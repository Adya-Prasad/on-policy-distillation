<h2 style="color: #fc9b23; border: 1px solid; padding: 10px 0px; text-align:center;"> ON POLICY DISTILLATION </h2>

1. Distill `Qwen/Qwen2.5-0.5B-Instruct` for Structured Tool-Use Syntax
2. Reproduces the continual-learning result from Thinking Machines' on-policy distillation.Demonstrating that supervised fine-tuning a Qwen2.5-0.5B model on a JSON task induces catastrophic forgetting of instruction-following (65% → 45%), which on-policy reverse-KL distillation against the model's own pre-SFT checkpoint then partially recovers (45% → 55%) while preserving the learned JSON skill

### Resources
1. On-Policy Distillation by _Kevin Lu in collaboration with others at Thinking Machines_ https://thinkingmachines.ai/blog/on-policy-distillation/
2. Unlocking On-Policy: Distillation for Any Model Family by _HuggingFace_ https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
3. Distilling 100B+ Models 40x Faster with TRL by _HuggingFace_ https://huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer
4. A Survey of On-Policy Distillation for Large Language Models by _Mingyang et al._ https://arxiv.org/pdf/2604.00626
5. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes _by Rishabh et al._ https://arxiv.org/pdf/2306.13649
6. On-Policy Knowledge Distillation _by emergentmind_ https://www.emergentmind.com/topics/on-policy-knowledge-distillation