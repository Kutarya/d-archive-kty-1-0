import math

import torch
from torch import nn
from torch.nn import functional as F

from config import model_config


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x, cache=None, use_cache=False):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.heads, self.head_dim)
                   .transpose(1, 2) for t in (q, k, v)]
        previous = 0
        if cache is not None:
            previous = cache[0].shape[2]
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        mask = None
        if previous and length > 1:
            queries = previous + torch.arange(length, device=x.device)
            keys = torch.arange(k.shape[2], device=x.device)
            mask = keys[None, :] <= queries[:, None]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                 is_causal=previous == 0)
        result = self.projection(attended.transpose(1, 2).contiguous()
                                 .reshape(batch, length, dim))
        return (result, (k, v)) if use_cache else result


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(dim, heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Linear(ffn_dim, dim))

    def forward(self, x, cache=None, use_cache=False):
        attended = self.attention(self.attention_norm(x), cache, use_cache)
        if use_cache:
            attended, cache = attended
        x = x + attended
        x = x + self.mlp(self.mlp_norm(x))
        return (x, cache) if use_cache else x


class KTYModel(nn.Module):
    supports_kv_cache = True

    def __init__(self, config=None):
        super().__init__()
        self.config = dict(model_config() if config is None else config)
        c = self.config
        if c["VOCAB_SIZE"] != 256:
            raise ValueError("Byte tokenizer için VOCAB_SIZE = 256 olmalı.")
        if any(not isinstance(c[k], int) or c[k] <= 0 for k in c):
            raise ValueError("Mimari değerleri pozitif tamsayı olmalı.")
        dim, heads = c["EMBEDDING_DIM"], c["NUM_HEADS"]
        if dim % heads:
            raise ValueError("EMBEDDING_DIM, NUM_HEADS'e tam bölünmeli.")
        self.context_size = c["CONTEXT_SIZE"]
        self.token_embedding = nn.Embedding(256, dim)
        self.position_embedding = nn.Embedding(self.context_size, dim)
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads, c["FFN_DIM"])
                                    for _ in range(c["NUM_LAYERS"]))
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, 256, bias=False)
        self.apply(self._initialize)
        residual_std = 0.02 / math.sqrt(2 * c["NUM_LAYERS"])
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight, std=residual_std)
            nn.init.normal_(block.mlp[2].weight, std=residual_std)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, past_key_values=None, use_cache=False):
        if past_key_values is not None and not use_cache:
            raise ValueError("Geçmiş KV değerleri için use_cache=True gerekli.")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("KV cache blok sayısı modelle eşleşmiyor.")
        offset = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if tokens.ndim != 2 or not 0 < tokens.shape[1] + offset <= self.context_size:
            raise ValueError("Girdi [B,T], 1 <= T <= CONTEXT_SIZE olmalı.")
        positions = offset + torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        caches = []
        for i, block in enumerate(self.blocks):
            old = None if past_key_values is None else past_key_values[i]
            result = block(x, old, use_cache)
            if use_cache:
                x, cache = result
                caches.append(cache)
            else:
                x = result
        logits = self.lm_head(self.final_norm(x))
        return (logits, tuple(caches)) if use_cache else logits


class LegacyKTYModel(nn.Module):
    """Eski byte checkpointlerini boyutlarını değiştirmeden yükler."""
    def __init__(self, state):
        super().__init__()
        vocab, dim = state["token_embedding.weight"].shape
        self.context_size = state["position_embedding.weight"].shape[0]
        self.token_embedding = nn.Embedding(vocab, dim)
        self.position_embedding = nn.Embedding(self.context_size, dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, vocab)
        if "causal_mask" in state:
            self.register_buffer("causal_mask", state["causal_mask"].clone())
        self.load_state_dict(state, strict=True)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        scores = self.query(x) @ self.key(x).transpose(-2, -1)
        scores = scores / math.sqrt(x.shape[-1])
        if hasattr(self, "causal_mask"):
            length = tokens.shape[1]
            scores = scores.masked_fill(~self.causal_mask[:length, :length], -torch.inf)
        return self.output(scores.softmax(dim=-1) @ self.value(x))


def load_model_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model = KTYModel(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.chat_format = checkpoint.get("chat_format", "legacy")
        model.reasoning_tag = checkpoint.get("reasoning_tag", "")
        model.end_marker = checkpoint.get("end_marker", "")
    elif "query.weight" in checkpoint:
        model = LegacyKTYModel(checkpoint)
    else:
        raise ValueError("Checkpoint biçimi tanınmadı; eski dosya değiştirilmedi.")
    return model.to(device), checkpoint
