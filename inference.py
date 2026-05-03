import os, sys, types, io, torch
import torch.nn.functional as F
import sentencepiece as spm

# Shim flash-attn-3 with PyTorch SDPA so train_gpt_decode imports cleanly
def _flash_attn_func(q, k, v, causal=True):
    H_q, H_kv = q.size(-2), k.size(-2)
    g = H_q // H_kv
    k = k.repeat_interleave(g, dim=-2)
    v = v.repeat_interleave(g, dim=-2)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal,
    ).transpose(1, 2)

_shim = types.ModuleType('flash_attn_interface')
_shim.flash_attn_func = _flash_attn_func
sys.modules['flash_attn_interface'] = _shim

# Match the env vars used during training
os.environ.setdefault('PARALLEL_RESIDUAL_START', '7')
os.environ.setdefault('LOOP_PHASE2_AT', '0.65')
os.environ.setdefault('HESSIAN_CLIP_LAMBDA', '0.175')
os.environ.setdefault('VOCAB_SIZE', '8192')

from train_gpt_decode import (
    GPT, Hyperparameters, restore_fp32_params,
    _decompress, dequantize_mixed,
)

# ----------------------------------------------------------------------
# Toggle: choose which model to load
# ----------------------------------------------------------------------
USE_QUANTIZED = False   # False -> load final_model.pt (the unquantized one)
# ----------------------------------------------------------------------

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"using device: {device}")

h = Hyperparameters()

# Build model in fp32 on CPU first (most MPS-friendly path)
model = GPT(h).float()
restore_fp32_params(model)

if USE_QUANTIZED:
    # Load the int6+brotli artifact and dequantize back to floats
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    with open('final_model.int6.ptz', 'rb') as f:
        quant_blob = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob, h.compressor)),
        map_location='cpu',
    )
    deq_state = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)
    model.load_state_dict(deq_state, strict=True)
    print("loaded quantized model from final_model.int6.ptz")
else:
    sd = torch.load('final_model.pt', map_location='cpu')
    model.load_state_dict(sd, strict=True)
    print("loaded unquantized model from final_model.pt")

model = model.to(device)
if h.num_loops > 0:
    model.activate_looping(2)
model.eval()

sp = spm.SentencePieceProcessor(model_file='fineweb_8192_bpe.model')

@torch.no_grad()
def generate(prompt, max_new_tokens=80, temperature=0.8, top_k=40):
    ids = sp.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model.forward_logits(tokens)[0, -1].float() / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=1)
    return sp.decode(tokens[0].tolist())

for prompt in [
    "The capital of France is",
    "Once upon a time in a small village,",
    "The most important thing about machine learning is",
]:
    print("---")
    print(generate(prompt))