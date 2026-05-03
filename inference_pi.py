"""
RUN WITH: OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python inference_pi.py

Inference for the train_gpt_decode model, tuned for Raspberry Pi 3 Model B.

Hardware target: Cortex-A53 4-core @ 1.2 GHz, 1 GB RAM, no usable GPU.
Loads `final_model.pt` (fp32). The quantized .ptz path is gone — both formats
end up as fp32 in RAM after loading, so there's no Pi-side benefit to .ptz.

Run with:
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python inference_pi.py
"""

import os
import sys
import types
import time
import gc

# Thread envs must be set before importing torch / numpy so BLAS picks them up.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")  # harmless if MKL absent

# Match the env vars used during training (must be set before importing the model).
os.environ.setdefault("PARALLEL_RESIDUAL_START", "7")
os.environ.setdefault("LOOP_PHASE2_AT", "0.65")
os.environ.setdefault("HESSIAN_CLIP_LAMBDA", "0.175")
os.environ.setdefault("VOCAB_SIZE", "8192")

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import sentencepiece as spm

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------------
# Flash-attn shim. Force the math backend so PyTorch doesn't probe alternatives
# on every call. On CPU there is no flash kernel anyway.
# ---------------------------------------------------------------------------
def _flash_attn_func(q, k, v, causal=True):
    H_q, H_kv = q.size(-2), k.size(-2)
    g = H_q // H_kv
    if g != 1:
        k = k.repeat_interleave(g, dim=-2)
        v = v.repeat_interleave(g, dim=-2)
    # (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)

_shim = types.ModuleType("flash_attn_interface")
_shim.flash_attn_func = _flash_attn_func
sys.modules["flash_attn_interface"] = _shim

from train_gpt_decode import GPT, Hyperparameters, restore_fp32_params

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOOP_PHASE      = 1      # 0 = no looping (fastest), 1 = one repeat, 2 = full (slowest).
                         # On Pi 3, phase 0 is the right default. Try 1 if quality
                         # feels off; phase 2 will roughly double per-token time.
GREEDY          = False  # True = argmax, skips topk + multinomial overhead.
DEFAULT_MAX_NEW = 60
DEFAULT_TEMP    = 0.8
DEFAULT_TOP_K   = 40
# Cap context to control attention cost. Trained at 2048; on a Pi don't go there.
MAX_CONTEXT     = 512

# Pre-warming the rotary cache makes the first generated token faster but
# briefly spikes RAM at startup. If load OOMs, set this to False.
PREWARM_ROTARY  = True

device = torch.device("cpu")
print(f"using device: {device} | threads={torch.get_num_threads()}")

# ---------------------------------------------------------------------------
# Build & load model
# ---------------------------------------------------------------------------
h = Hyperparameters()

t0 = time.time()
model = GPT(h).float()
restore_fp32_params(model)

# Load weights, then immediately drop the state dict reference so we don't keep
# a duplicate copy of the weights in memory during the GC cycle.
sd = torch.load("final_model.pt", map_location="cpu")
model.load_state_dict(sd, strict=True)
del sd
gc.collect()
print(f"loaded final_model.pt in {time.time() - t0:.1f}s")

if h.num_loops > 0:
    model.activate_looping(LOOP_PHASE)
    print(f"looping phase: {LOOP_PHASE}")

model.eval()

if PREWARM_ROTARY:
    # Prime each block's rotary cos/sin cache once so the first generation step
    # doesn't pay for it. Uses a throwaway forward at MAX_CONTEXT.
    with torch.inference_mode():
        dummy = torch.zeros((1, MAX_CONTEXT), dtype=torch.long, device=device)
        _ = model.forward_logits(dummy)
        del dummy
    gc.collect()

print(f"model ready in {time.time() - t0:.1f}s")

sp = spm.SentencePieceProcessor(model_file="fineweb_8192_bpe.model")

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate(prompt: str,
             max_new_tokens: int = DEFAULT_MAX_NEW,
             temperature: float = DEFAULT_TEMP,
             top_k: int = DEFAULT_TOP_K,
             greedy: bool = GREEDY,
             stream: bool = True,
             verbose: bool = True) -> str:
    ids = sp.encode(prompt)
    budget = MAX_CONTEXT - max_new_tokens
    if len(ids) > budget:
        ids = ids[-budget:]
    prefix_len = len(ids)
    total_len  = prefix_len + max_new_tokens

    tokens = torch.zeros((1, total_len), dtype=torch.long, device=device)
    tokens[0, :prefix_len] = torch.tensor(ids, dtype=torch.long)

    if stream:
        print(sp.decode(ids), end="", flush=True)

    pos = prefix_len
    pending = b""  # buffer for incomplete UTF-8 from byte-fallback tokens
    t_start = time.time()
    t_first = None

    for step in range(max_new_tokens):
        logits = model.forward_logits(tokens[:, :pos])[0, -1]

        if greedy:
            next_tok = int(torch.argmax(logits).item())
        else:
            logits = logits.float() / temperature
            if 0 < top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits = torch.where(
                    logits < v[-1],
                    torch.full_like(logits, float("-inf")),
                    logits,
                )
            probs = torch.softmax(logits, dim=-1)
            next_tok = int(torch.multinomial(probs, 1).item())

        tokens[0, pos] = next_tok
        pos += 1

        if step == 0:
            t_first = time.time() - t_start

        if stream:
            piece = sp.id_to_piece(next_tok)
            # Byte-fallback tokens look like '<0xE2>' — collect bytes until we
            # have a complete UTF-8 codepoint, then emit.
            if piece.startswith("<0x") and piece.endswith(">"):
                pending += bytes([int(piece[3:-1], 16)])
                try:
                    chunk = pending.decode("utf-8")
                    pending = b""
                    print(chunk, end="", flush=True)
                except UnicodeDecodeError:
                    pass  # wait for more bytes
            else:
                # Normal piece. ▁ (U+2581) is the SentencePiece word-start marker.
                if pending:
                    # Flush stray bytes as replacement chars rather than losing them.
                    print(pending.decode("utf-8", errors="replace"), end="", flush=True)
                    pending = b""
                print(piece.replace("\u2581", " "), end="", flush=True)

    elapsed = time.time() - t_start
    if stream:
        if pending:
            print(pending.decode("utf-8", errors="replace"), end="", flush=True)
        print()
    if verbose:
        tps = max_new_tokens / elapsed if elapsed > 0 else 0
        print(f"  [first token: {t_first:.2f}s | {max_new_tokens} tokens in {elapsed:.1f}s = {tps:.2f} tok/s]")

    return sp.decode(tokens[0, :pos].tolist())


if __name__ == "__main__":
    for prompt in [
        "The capital of France is",
        "Once upon a time in a small village,",
        "The most important thing about machine learning is",
    ]:
        print("---")
        generate(prompt)