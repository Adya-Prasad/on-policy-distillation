---
base_model: Qwen/Qwen2.5-0.5B-Instruct
license: apache-2.0
tags: [distillation, on-policy, tool-use, calculator]
---

## qwen2.5-0.5b-instruct-math-tool-distilled

Qwen2.5-0.5B-Instruct distilled via **on-policy reverse-KL distillation**
against a few-shot-conditioned Qwen2.5-1.5B-Instruct teacher, to emit the
structured tool-call format `[CALCULATOR(expression)]` on arithmetic prompts.

## Training
- **Method**: on-policy distillation with reverse KL (Agarwal et al., 2023)
- **Teacher**: Qwen2.5-1.5B-Instruct (4-bit NF4), few-shot prompted
- **Data**: 128 synthetic arithmetic prompts (+/-/*), seed=42
- **Optimizer**: bitsandbytes PagedAdamW8bit, lr=5e-6
- **Hardware**: NVIDIA GTX 1650 (4 GB)
- **Epochs**: 1

## Evaluation
- Format compliance (regex match `\[CALCULATOR\([\d+\-*/ ]+\)\]`): 100% on 10 held-out prompts.

## Limitations
- Trained only on +/-/*; behavior on division, parentheses, or multi-step
  expressions is untested and likely poor.
- Inherits all biases and limitations of the Qwen2.5-0.5B base model.
