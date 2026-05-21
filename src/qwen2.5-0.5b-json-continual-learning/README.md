---
base_model: Qwen/Qwen2.5-0.5B-Instruct
license: apache-2.0
tags: [continual-learning, on-policy-distillation, self-distillation]
---

# qwen2.5-0.5b-json-continual-v1

Reproduction of Thinking Machines' continual-learning experiment
("On-Policy Distillation", Lu et al., Oct 2025) at consumer scale.

## Method
1. SFT on synthetic JSON-emission task (800 examples, 1 epoch) →
   catastrophic forgetting of instruction-following.
2. On-policy reverse-KL distillation against the original Qwen2.5-0.5B-Instruct
   as teacher, on general instruction prompts (80 prompts, 2 epochs) →
   instruction-following recovered.

## Results
| Stage      | IF score | JSON score |
|------------|----------|------------|
| Baseline   | 65.0%  | 66.7%   |
| Post-SFT   | 45.0%  | 75.0%   |
| Post-OPD   | 55.0%  | 75.0%   |

## Hardware
NVIDIA GTX 1650 (4 GB), Windows + PowerShell.
Student in FP16, teacher in 4-bit NF4, optimizer = bitsandbytes PagedAdamW8bit.
