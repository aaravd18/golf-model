"""
RUN WITH: OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python inference_pi.py

Inference for the train_gpt_decode model, tuned for Raspberry Pi 3 Model B.
Hardware target: Cortex-A53 4-core @ 1.2 GHz, 1 GB RAM, no usable GPU.

Prefers `final_model.int6.ptz` (quantized) if present, otherwise falls back
to `final_model.pt` (fp32). Both end up as fp32 in RAM after loading; the
.ptz is just smaller on disk and slightly noisier in output quality.

Run with:
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python inference_pi.py
"""
import os
import sys
import io
import types
import time
import gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4") 

# Match the env vars used during training
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

from train_gpt_decode import (
    GPT, Hyperparameters, restore_fp32_params,
    _decompress, dequantize_mixed,
)

# Config
QUANT_PATH      = "final_model.int6.ptz"
FP32_PATH       = "final_model.pt"

LOOP_PHASE      = 1      # 0 = no looping (fastest), 1 = one repeat, 2 = full (slowest).
GREEDY          = False  
DEFAULT_MAX_NEW = 60
DEFAULT_TEMP    = 0.8
DEFAULT_TOP_K   = 40
# Context trained at 2048 but capped at 512 for pi
MAX_CONTEXT     = 512
PREWARM_ROTARY  = True

device = torch.device("cpu")
print(f"using device: {device} | threads={torch.get_num_threads()}")

# ---------------------------------------------------------------------------
# Build & load model — prefer quantized, fall back to fp32
# ---------------------------------------------------------------------------
h = Hyperparameters()
t0 = time.time()
model = GPT(h).float()
restore_fp32_params(model)

if os.path.exists(QUANT_PATH):
    file_kb = os.path.getsize(QUANT_PATH) / 1024
    print(f"loading quantized model: {QUANT_PATH} ({file_kb:.0f} KB)")
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    with open(QUANT_PATH, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob, h.compressor)),
        map_location="cpu",
    )
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)
    model.load_state_dict(deq_state, strict=True)
    del sd_cpu, quant_blob, quant_state, deq_state
elif os.path.exists(FP32_PATH):
    file_mb = os.path.getsize(FP32_PATH) / (1024 * 1024)
    print(f"loading fp32 model: {FP32_PATH} ({file_mb:.0f} MB)")
    sd = torch.load(FP32_PATH, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    del sd
else:
    raise FileNotFoundError(
        f"neither {QUANT_PATH} nor {FP32_PATH} found in current directory"
    )

gc.collect()
print(f"weights loaded in {time.time() - t0:.1f}s")

if h.num_loops > 0:
    model.activate_looping(LOOP_PHASE)
    print(f"looping phase: {LOOP_PHASE}")
model.eval()

if PREWARM_ROTARY:
    # Prime each block's rotary cos/sin cache once so the first generation step
    # doesn't pay for it
    with torch.inference_mode():
        dummy = torch.zeros((1, MAX_CONTEXT), dtype=torch.long, device=device)
        _ = model.forward_logits(dummy)
        del dummy
    gc.collect()

print(f"model ready in {time.time() - t0:.1f}s")

sp = spm.SentencePieceProcessor(model_file="fineweb_8192_bpe.model")

# Streamed generation
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