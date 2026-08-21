"""Standalone port of SGLang's LlamaForCausalLMEagle3 draft-head layer math,
shared by scripts/probe_draft_routing_predictability.py and
scripts/train_routing_proxy.py so both use identical, single-source feature
extraction -- avoids the two scripts silently drifting into different
(and differently wrong) implementations of the same forward pass.

Ported line-for-line from installed sglang 0.5.17
(sglang/srt/models/llama_eagle3.py's LlamaDecoderLayer.forward,
LlamaModel.forward; sglang/srt/models/llama.py's
LlamaAttention.forward_prepare_native), reading real checkpoint weights via
safetensors rather than depending on sglang's ForwardBatch/spec_info serving
plumbing to run one teacher-forced forward pass. See
scripts/probe_draft_routing_predictability.py's original docstring for the
full trace of why the checkpoint isn't loadable via plain
AutoModelForCausalLM (no embed_tokens, single `midlayer` not a `layers.N`
list, `LlamaForCausalLMEagle3` only exists inside sglang's own source).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x.to(dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor,
               head_dim: int, rope_theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Neox-style RoPE (get_rope(..., is_neox_style=True) in sglang's
    LlamaAttention, the standard HF Llama convention: full head_dim rotary,
    rotate_half on interleaved-by-half (not interleaved-by-pair) layout."""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)  # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)            # (T, head_dim)
    cos = emb.cos()[None, :, None, :].to(q.dtype)       # (1, T, 1, head_dim)
    sin = emb.sin()[None, :, None, :].to(q.dtype)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class Eagle3DraftLayer(nn.Module):
    """See module docstring. Mirrors, in order:
      - LlamaModel.forward: hidden_states = fc(concat(aux_layers)) [no
        input_norm/fc_norm -- this checkpoint's config has neither
        norm_before_fc nor fc_norm/use_aux_norm set, confirmed by the
        absence of input_norm.*/fc_norm.* tensors in its safetensors file].
      - LlamaDecoderLayer.forward (is_input_layer=True branch, the only
        branch that matters since num_hidden_layers=1):
            residual = hidden_states
            hidden_states = hidden_norm(hidden_states)
            embeds = input_layernorm(embeds)
            hidden_states = cat([embeds, hidden_states], dim=-1)
            hidden_states = self_attn(hidden_states)   # qkv_proj takes 2*hidden_size in
            hidden_states, residual = post_attention_layernorm(hidden_states, residual)
            hidden_states = mlp(hidden_states)
      - LlamaAttention.forward_prepare_native: qkv_proj -> split -> RoPE ->
        scaled-dot-product attention (RadixAttention reduces to plain
        causal SDPA for a single non-paged forward) -> o_proj.
      - The final `self.norm(hidden_states, residual)` (RMSNorm with a
        fused residual add, matching sglang's RMSNorm(x, residual) API:
        returns (norm(x + residual), x + residual)) is applied here too,
        as the layer's own output (matches LlamaModel.forward calling
        self.norm right after the single decoder layer).
    """

    def __init__(self, weights: dict, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, rope_theta: float, rms_norm_eps: float, device: str, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.eps = rms_norm_eps
        self.scaling = head_dim ** -0.5

        def w(name):
            return weights[name].to(device=device, dtype=dtype)

        self.input_layernorm_w = w("midlayer.input_layernorm.weight")
        self.hidden_norm_w = w("midlayer.hidden_norm.weight")
        self.post_attn_norm_w = w("midlayer.post_attention_layernorm.weight")
        self.q_proj_w = w("midlayer.self_attn.q_proj.weight")
        self.k_proj_w = w("midlayer.self_attn.k_proj.weight")
        self.v_proj_w = w("midlayer.self_attn.v_proj.weight")
        self.o_proj_w = w("midlayer.self_attn.o_proj.weight")
        self.gate_proj_w = w("midlayer.mlp.gate_proj.weight")
        self.up_proj_w = w("midlayer.mlp.up_proj.weight")
        self.down_proj_w = w("midlayer.mlp.down_proj.weight")
        self.final_norm_w = w("norm.weight")
        self.fc_w = w("fc.weight")

    def forward(self, embeds: torch.Tensor, aux_concat: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # embeds, aux_concat: (T, hidden_size) / (T, hidden_size_in*num_aux); positions: (T,)
        hidden_states = F.linear(aux_concat, self.fc_w)              # (T, hidden_size)
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.hidden_norm_w, self.eps)
        embeds_n = rms_norm(embeds, self.input_layernorm_w, self.eps)
        hidden_states = torch.cat([embeds_n, hidden_states], dim=-1)  # (T, 2*hidden_size)

        T = hidden_states.shape[0]
        q = F.linear(hidden_states, self.q_proj_w).view(T, self.num_heads, self.head_dim)
        k = F.linear(hidden_states, self.k_proj_w).view(T, self.num_kv_heads, self.head_dim)
        v = F.linear(hidden_states, self.v_proj_w).view(T, self.num_kv_heads, self.head_dim)

        q, k = apply_rope(q.unsqueeze(0), k.unsqueeze(0), positions, self.head_dim, self.rope_theta)
        q, k = q.squeeze(0), k.squeeze(0)

        n_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        q_ = q.transpose(0, 1).unsqueeze(0)  # (1, num_heads, T, head_dim)
        k_ = k.transpose(0, 1).unsqueeze(0)
        v_ = v.transpose(0, 1).unsqueeze(0)
        attn_out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True, scale=self.scaling)
        attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(T, self.num_heads * self.head_dim)

        hidden_states = F.linear(attn_out, self.o_proj_w)  # (T, hidden_size)

        # post_attention_layernorm fuses residual: norm(x + residual), then residual := x + residual
        combined = hidden_states.float() + residual.float()
        hidden_states = rms_norm(combined.to(hidden_states.dtype), self.post_attn_norm_w, self.eps)
        residual = combined.to(hidden_states.dtype)

        gate = F.linear(hidden_states, self.gate_proj_w)
        up = F.linear(hidden_states, self.up_proj_w)
        mlp_out = F.linear(F.silu(gate) * up, self.down_proj_w)

        # final self.norm(hidden_states, residual) with fused residual add
        final_combined = mlp_out.float() + residual.float()
        normed = rms_norm(final_combined.to(mlp_out.dtype), self.final_norm_w, self.eps)
        return normed  # (T, hidden_size) -- this IS the draft hidden state


def get_target_aux_layer_ids(target_config) -> list[int]:
    """EAGLE3 draft heads are trained on a fixed 3-layer aux-hidden-state
    recipe (low/mid/high depth), the same convention SpecForge and SGLang's
    own EAGLE3 capture use: roughly {2, L//2, L-3} for an L-layer target,
    0-indexed into hidden_states (index 0 is the embedding output, so
    decoder layer i's output is hidden_states[i+1]).
    Confirmed pattern, not re-derived per-model: SGLang's
    `set_eagle3_layers_to_capture` (referenced in AXIS8.md's compatibility
    check) uses this same low/mid/high spread. If results look suspiciously
    flat, re-verify this triple against the draft checkpoint's own training
    config before trusting a null result.
    """
    L = target_config.num_hidden_layers
    return sorted(set([2, L // 2, L - 3]))


def load_eagle3_draft_layer(draft_model_repo: str, device: str = "cuda",
                            dtype: torch.dtype = torch.bfloat16) -> tuple[Eagle3DraftLayer, int]:
    """Loads the draft checkpoint's weights and returns (layer, hidden_size).
    Raises loudly (not a silent fallback) if the checkpoint doesn't match
    the expected sglang 0.5.17 llama_eagle3.py tensor layout -- see
    probe_draft_routing_predictability.py's original commit message for why
    a silent fallback here would produce a meaningless result that looks
    like, but isn't, a real signal."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoConfig

    draft_config = AutoConfig.from_pretrained(draft_model_repo)
    draft_hidden_size = draft_config.hidden_size

    draft_weights_path = hf_hub_download(repo_id=draft_model_repo, filename="model.safetensors")
    draft_weights = load_safetensors(draft_weights_path)
    expected = {"fc.weight", "midlayer.input_layernorm.weight", "midlayer.hidden_norm.weight",
                "midlayer.self_attn.q_proj.weight", "norm.weight"}
    missing = expected - set(draft_weights.keys())
    if missing:
        raise RuntimeError(
            f"draft checkpoint is missing expected tensors {missing} -- this "
            f"port assumes SGLang 0.5.17's llama_eagle3.py layout (fc + "
            f"midlayer.* + norm). Checkpoint's actual tensors: "
            f"{sorted(draft_weights.keys())}. Do not proceed with a "
            f"mismatched port; re-check against the installed sglang "
            f"source before rerunning."
        )

    rope_parameters = getattr(draft_config, "rope_parameters", None)
    if rope_parameters is not None:
        rope_theta = rope_parameters.get("rope_theta", 10000)
    else:
        rope_theta = getattr(draft_config, "rope_theta", 10000)

    layer = Eagle3DraftLayer(
        draft_weights,
        hidden_size=draft_hidden_size,
        num_heads=draft_config.num_attention_heads,
        num_kv_heads=draft_config.num_key_value_heads,
        head_dim=getattr(draft_config, "head_dim", draft_hidden_size // draft_config.num_attention_heads),
        rope_theta=rope_theta,
        rms_norm_eps=draft_config.rms_norm_eps,
        device=device, dtype=dtype,
    )
    return layer, draft_hidden_size
