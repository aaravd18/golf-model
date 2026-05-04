# Running a 16MB Model on a Raspberry Pi

This repo demonstrates that **compressed intelligence can run on low-power edge hardware**. Specifically, a 16MB quantized language model running inference on a Raspberry Pi 3. The goal is to show that with aggressive model compression and efficient training techniques, capable models need not be confined to data centers or high-end GPUs.

The training script and inference code here are built around the constraints of the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition, which serves as a forcing function for maximizing model quality within tight size and compute budgets.

## Credits

This work is based on the entry [`2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence`](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence) submitted to the **10-minute / 16MB track** of the OpenAI Parameter Golf leaderboard.

## What This Repo Is

- **Training**: A script to train a 16MB model using the techniques from the entry above (incorporates parallel residuals, depth recurrence, hessian-aware quantization).
- **Inference**: [`inference_pi.py`](./inference_pi.py) contains code to run the 16MB quantized model on a **Raspberry Pi 3**, demonstrating that a competitively trained small model can perform inference on low-power edge hardware.

## Training

To train the model, make the training script executable and run it:

```bash
chmod +x train.sh
./train.sh
```

Training was run on an **8x H100 node for 10 minutes**, in accordance with the Parameter Golf competition constraints.
