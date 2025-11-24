
import torch
import pickle
import os
import io
import argparse
import gc
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import save_file
from tqdm import tqdm

# ==========================================
# Mock Classes for Custom Pickle Types
# ==========================================

class MetadataIndex:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __repr__(self):
        return f"MetadataIndex({self.__dict__})"
    def __hash__(self):
        return hash((self.__dict__.get('index'), self.__dict__.get('fqn')))
    def __eq__(self, other):
        return isinstance(other, MetadataIndex) and self.__dict__ == other.__dict__

@dataclass
class _StoragePrefix:
    prefix: str

@dataclass
class _StorageInfo:
    relative_path: str
    offset: int
    length: int
    transform_descriptors: Any = None

class ChunkStorageMetadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __repr__(self):
        return f"ChunkStorageMetadata({self.__dict__})"

class TensorStorageMetadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __repr__(self):
        return f"TensorStorageMetadata({self.__dict__})"

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'MetadataIndex': return MetadataIndex
        if name == '_StoragePrefix': return _StoragePrefix
        if name == '_StorageInfo': return _StorageInfo
        if name == 'ChunkStorageMetadata': return ChunkStorageMetadata
        if name == 'TensorStorageMetadata': return TensorStorageMetadata
        return super().find_class(module, name)

# ==========================================
# Custom Loader Logic
# ==========================================

class CustomCheckpointLoader:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.shard_metadata_cache = {} # path -> metadata dict
        self.shards = self._discover_shards()
        print(f"Discovered {len(self.shards)} shards.")

    def _discover_shards(self):
        shards = [d for d in os.listdir(self.checkpoint_dir) if os.path.isdir(os.path.join(self.checkpoint_dir, d)) and "tp" in d]
        shards.sort()
        return shards

    def _load_chunk_metadata(self, shard_path, chunk_idx):
        cache_key = (shard_path, chunk_idx)
        if cache_key in self.shard_metadata_cache:
            return self.shard_metadata_cache[cache_key]
            
        meta_path = os.path.join(shard_path, f".__{chunk_idx}_.metadata")
        if not os.path.exists(meta_path):
            return {}
            
        with open(meta_path, "rb") as f:
            data = CustomUnpickler(f).load()
            if hasattr(data, 'storage_data'):
                self.shard_metadata_cache[cache_key] = data.storage_data
                return data.storage_data
            return {}

    def load_tensor_shards(self, fqn):
        """
        Loads all chunks for a given FQN from all shards.
        Returns a list of tensors (one per TP rank, ordered by rank).
        Each TP rank tensor is assembled from its internal chunks.
        """
        tp_shards = []
        found_any = False
        
        # Iterate over TP ranks (tp0, tp1, ...)
        for shard in self.shards:
            shard_path = os.path.join(self.checkpoint_dir, shard)
            
            # Load all chunks for this TP rank
            rank_chunks = {} # chunk_idx -> tensor
            
            # We assume up to 20 chunks to be safe
            for chunk_idx in range(20):
                chunk_meta = self._load_chunk_metadata(shard_path, chunk_idx)
                if not chunk_meta:
                    continue
                
                # Search for FQN
                for key_obj, storage_info in chunk_meta.items():
                    if hasattr(key_obj, 'fqn') and key_obj.fqn == fqn:
                        data_file = os.path.join(shard_path, storage_info.relative_path)
                        chunk_tensor = self._read_tensor_from_file(data_file, storage_info.offset, storage_info.length)
                        if chunk_tensor is not None:
                            rank_chunks[chunk_idx] = chunk_tensor
                            found_any = True
                        break
            
            if rank_chunks:
                # Concatenate chunks for this rank
                sorted_indices = sorted(rank_chunks.keys())
                sorted_tensors = [rank_chunks[i] for i in sorted_indices]
                
                try:
                    tp_shard_tensor = torch.cat(sorted_tensors, dim=0)
                    tp_shards.append(tp_shard_tensor)
                except Exception as e:
                    print(f"Error concatenating chunks for {fqn} in {shard}: {e}")
            else:
                pass
        
        if not found_any:
            return None
            
        return tp_shards

    def _read_tensor_from_file(self, filepath, offset, length):
        try:
            with open(filepath, "rb") as f:
                f.seek(offset)
                data_bytes = f.read(length)
            buffer = io.BytesIO(data_bytes)
            return torch.load(buffer, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Failed to read/deserialize tensor from {filepath}: {e}")
            return None

# ==========================================
# Conversion Logic
# ==========================================

LLAMA3_70B_CONFIG = LlamaConfig(
    vocab_size=128256,
    hidden_size=8192,
    intermediate_size=28672,
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=8192,
    initializer_range=0.02,
    rms_norm_eps=1e-05,
    use_cache=True,
    pad_token_id=128001,
    bos_token_id=128000,
    eos_token_id=128001,
    tie_word_embeddings=False,
    rope_theta=500000.0,
    attention_bias=False,
    attention_dropout=0.0,
)

def convert_checkpoint(base_path, output_path):
    print(f"Starting conversion for {base_path}")
    print(f"Target output: {output_path}")
    print("Configuration: RoPE Permutation=False, MLP Order=Gate/Up")
    
    os.makedirs(output_path, exist_ok=True)
    
    loader = CustomCheckpointLoader(base_path)
    hf_state_dict = {}
    missing_tensors = []
    
    # Helper params
    q_dim = 1024
    k_dim = 128
    v_dim = 128
    gate_dim = 3584
    up_dim = 3584
    
    # --- Embeddings ---
    print("Processing Embeddings...")
    emb_key = "state_dict.word_embedding.weight"
    chunks = loader.load_tensor_shards(emb_key)
    if chunks:
        hf_state_dict['model.embed_tokens.weight'] = torch.cat(chunks, dim=0).to(torch.bfloat16)
    else:
        missing_tensors.append("embeddings")

    # --- Layers ---
    total_layers = LLAMA3_70B_CONFIG.num_hidden_layers
    print(f"Processing {total_layers} layers...")
    
    for i in tqdm(range(total_layers)):
        prefix = f"state_dict.transformer.seq_layers.{i}.layer"
        
        # 1. Input LayerNorm
        ln1_key = f"{prefix}.self_attention.layernorm_qkv.layer_norm_weight"
        chunks = loader.load_tensor_shards(ln1_key)
        if chunks:
            hf_state_dict[f'model.layers.{i}.input_layernorm.weight'] = chunks[0].to(torch.bfloat16)
        else:
            missing_tensors.append(f"layer_{i}_ln1")

        # 2. QKV Proj
        qkv_key = f"{prefix}.self_attention.layernorm_qkv.weight"
        chunks = loader.load_tensor_shards(qkv_key)
        if chunks:
            qs, ks, vs = [], [], []
            for w in chunks:
                w = w.to(torch.bfloat16)
                try:
                    # Standard QKV split
                    q, k, v = torch.split(w, [q_dim, k_dim, v_dim], dim=0)
                    
                    # No RoPE permutation needed for this checkpoint format
                    qs.append(q)
                    ks.append(k)
                    vs.append(v)
                except RuntimeError as e:
                    print(f"Error splitting QKV for layer {i}: {e}")
                    missing_tensors.append(f"layer_{i}_qkv_SHAPE_MISMATCH")
                    break
            
            if qs:
                hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = torch.cat(qs, dim=0)
                hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = torch.cat(ks, dim=0)
                hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = torch.cat(vs, dim=0)
        else:
            missing_tensors.append(f"layer_{i}_qkv")

        # 3. Output Proj
        o_key = f"{prefix}.self_attention.proj.weight"
        chunks = loader.load_tensor_shards(o_key)
        if chunks:
            hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = torch.cat(chunks, dim=1).to(torch.bfloat16)
        else:
            missing_tensors.append(f"layer_{i}_o_proj")

        # 4. Post Attention LayerNorm
        ln2_key = f"{prefix}.layernorm_mlp.layer_norm_weight"
        chunks = loader.load_tensor_shards(ln2_key)
        if chunks:
            hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = chunks[0].to(torch.bfloat16)
        else:
            missing_tensors.append(f"layer_{i}_ln2")

        # 5. MLP Gate/Up
        fc1_key = f"{prefix}.layernorm_mlp.fc1_weight"
        chunks = loader.load_tensor_shards(fc1_key)
        if chunks:
            gs, us = [], []
            for w in chunks:
                w = w.to(torch.bfloat16)
                # Standard Gate/Up split
                g, u = torch.split(w, [gate_dim, up_dim], dim=0)
                gs.append(g)
                us.append(u)
            hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = torch.cat(gs, dim=0)
            hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = torch.cat(us, dim=0)
        else:
            missing_tensors.append(f"layer_{i}_mlp_fc1")

        # 6. MLP Down
        fc2_key = f"{prefix}.layernorm_mlp.fc2_weight"
        chunks = loader.load_tensor_shards(fc2_key)
        if chunks:
            hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = torch.cat(chunks, dim=1).to(torch.bfloat16)
        else:
            missing_tensors.append(f"layer_{i}_mlp_fc2")

    # --- Final Norm ---
    ln_f_key = "state_dict.transformer.final_layernorm.weight"
    chunks = loader.load_tensor_shards(ln_f_key)
    if not chunks:
        ln_f_key = "state_dict.layernorm.weight"
        chunks = loader.load_tensor_shards(ln_f_key)

    if chunks:
        hf_state_dict['model.norm.weight'] = chunks[0].to(torch.bfloat16)
    else:
        missing_tensors.append("final_norm")

    # --- LM Head ---
    lm_head_key = "state_dict.lm_head.weight"
    chunks = loader.load_tensor_shards(lm_head_key)
    if not chunks:
        lm_head_key = "state_dict.output_layer.weight"
        chunks = loader.load_tensor_shards(lm_head_key)
        
    if chunks:
        hf_state_dict['lm_head.weight'] = torch.cat(chunks, dim=0).to(torch.bfloat16)
    else:
        missing_tensors.append("lm_head")

    # Report
    if missing_tensors:
        print(f"⚠️  WARNING: {len(missing_tensors)} MISSING TENSORS!")
        print(f"Sample: {missing_tensors[:5]}")
    else:
        print("✅ All tensors found.")

    # Save
    print("Saving to disk...")
    LLAMA3_70B_CONFIG.save_pretrained(output_path)
    
    try:
        model = LlamaForCausalLM(LLAMA3_70B_CONFIG)
        model.load_state_dict(hf_state_dict, strict=False)
        model.to(torch.bfloat16)
        model.save_pretrained(output_path, safe_serialization=True, max_shard_size="10GB")
        print("✅ Conversion complete!")
    except Exception as e:
        print(f"❌ Save failed: {e}")
        save_file(hf_state_dict, os.path.join(output_path, "model.safetensors"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/workspace1/pre-training-latamgpt/model-checkpoints/cpt-llama370b-fp8-fl-arrow-pack-14-14fsdp-tp8-bs3/step_52000")
    parser.add_argument("--output_path", type=str, default="/workspace1/gonzalo.fuentes/checkpoint_converter/converted_llama3_final")
    
    args = parser.parse_args()
    convert_checkpoint(args.input_path, args.output_path)
