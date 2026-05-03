# Parameter Golf — 16MB Model on Raspberry Pi
 
This repo contains the training script and inference code for a 16MB language model trained under the constraints of the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition.
 
## Credits
 
This work is based on the entry [`2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence`](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence) submitted to the **10-minute / 16MB track** of the OpenAI Parameter Golf leaderboard.
 
## What This Repo Is
 
- **Training**: A script to train a 16MB model using the techniques from the entry above (SP8192 sequence packing, Hessian-scaled gradient clipping, and progressive recurrence).
- **Inference**: Code to run the trained model on a **Raspberry Pi**, demonstrating that a competitively trained small model can perform inference on low-power edge hardware.
## Highlights
 
- Model size: ≤ 16MB
- Training time constraint: 10 minutes
- Inference target: Raspberry Pi (ARM, CPU-only)
