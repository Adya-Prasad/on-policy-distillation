# SUMMARY NOTE

## (1) KL Divergence

Kullback-Leibler (KL) Divergence measures the asymmetric statistical distance between two probability distributions over the same support. In LLM distillation, we use it to measure how much the student’s token-level probability distribution ($\pi_\theta$) deviates from the teacher's target distribution ($\pi_{\text{teacher}}$).
### The Mathematical Formulation
For a single state $s_t$ (the text context at step $t$), the KL divergence from the student distribution to the teacher distribution is defined as:
$$\mathbb{D}_{\text{KL}} \left( \pi_{\text{teacher}}(\cdot \mid s_t) \parallel \pi_\theta(\cdot \mid s_t) \right) = \sum_{w \in \mathcal{V}} \pi_{\text{teacher}}(w \mid s_t) \cdot \log \left( \frac{\pi_{\text{teacher}}(w \mid s_t)}{\pi_\theta(w \mid s_t)} \right)$$ 
### Dissecting the Symbols

* $\mathcal{V}$: The entire vocabulary dictionary (the action space). Every valid token $w$ is evaluated.
* $\pi_{\text{teacher}}(w \mid s_t)$: The exact probability the teacher assigns to token $w$ given the context $s_t$.
* $\pi_\theta(w \mid s_t)$: The exact probability the student assigns to token $w$ given the context $s_t$.
* $\log \left( \frac{\pi_{\text{teacher}}}{\pi_\theta} \right)$: The log ratio of their beliefs. Using logarithm rules, this can be expanded into:
$$\log \pi_{\text{teacher}}(w \mid s_t) - \log \pi_\theta(w \mid s_t)$$ 

### Expanding the Equation for Optimization
Let's distribute the summation across the log subtraction to see what the neural network actually optimizes:
$$\mathbb{D}_{\text{KL}} = \sum_{w \in \mathcal{V}} \pi_{\text{teacher}}(w \mid s_t) \log \pi_{\text{teacher}}(w \mid s_t) - \sum_{w \in \mathcal{V}} \pi_{\text{teacher}}(w \mid s_t) \log \pi_\theta(w \mid s_t)$$ 
Notice that the first term relies completely on the teacher. Because the teacher's weights are frozen during distillation, this term is a constant. We can rewrite the divergence as:
$$\mathbb{D}_{\text{KL}} = \mathcal{H}(\pi_{\text{teacher}}) + \mathcal{H}(\pi_{\text{teacher}}, \pi_\theta)$$ 

* $\mathcal{H}(\pi_{\text{teacher}})$: The negative entropy of the teacher (a constant scalar).
* $\mathcal{H}(\pi_{\text{teacher}}, \pi_\theta)$: The Cross-Entropy Loss between the teacher and student.


1. KL Divergence is a measure of information loss. If you use the student's probability distribution ($\pi_\theta$) to approximate the true teacher's distribution ($\pi_{\text{teacher}}$), KL Divergence calculates exactly how many bits of information you waste.
2. The Directional Behavior (Asymmetry): It is not a true distance metric because $\mathbb{D}_{\text{KL}}(P \parallel Q) \neq \mathbb{D}_{\text{KL}}(Q \parallel P)$.
    * Forward KL ($\mathbb{D}_{\text{KL}}(\text{Teacher} \parallel \text{Student})$) forces the student to cover all the modes (options) the teacher likes. It is mean-seeking.
    * Reverse KL ($\mathbb{D}_{\text{KL}}(\text{Student} \parallel \text{Teacher})$) forces the student to pick just one mode it is confident in and stay there. It is mode-seeking.
3. The Value Boundaries: It is always greater than or equal to zero ($\mathbb{D}_{\text{KL}} \ge 0$). It equals exactly $0$ if and only if the student and teacher distributions are identical.

### The Mathematical Computation 
Let's assume our vocabulary has only 3 tokens: ["blue", "cloudy", "banana"].
The model is processing the state: "The sky is..."
### Step 1: Extract Raw Teacher and Student Logits
The models output unnormalized scores called logits ($z$).

* Teacher Logits ($z_{\text{teacher}}$): [10.0, 6.0, -2.0]
* Student Logits ($z_\theta$): [3.0, 2.0, 1.0]

### Step 2: Convert Logits to Probability Distributions via Softmax
We turn logits into probabilities using the Softmax function: $P_i = \frac{e^{z_i}}{\sum e^{z_j}}$.

* Teacher Distribution ($\pi_{\text{teacher}}$):
$$\text{blue} = \frac{e^{10}}{e^{10}+e^6+e^{-2}} \approx \mathbf{0.98}$$ $$\text{cloudy} = \frac{e^6}{e^{10}+e^6+e^{-2}} \approx \mathbf{0.02}$$ $$\text{banana} = \frac{e^{-2}}{e^{10}+e^6+e^{-2}} \approx \mathbf{0.00}$$ $$\pi_{\text{teacher}} = [0.98, 0.02, 0.00]$$ 
* Student Distribution ($\pi_\theta$):
$$\text{blue} = \frac{e^3}{e^3+e^2+e^1} \approx \mathbf{0.67}$$ $$\text{cloudy} = \frac{e^2}{e^3+e^2+e^1} \approx \mathbf{0.24}$$ $$\text{banana} = \frac{e^1}{e^3+e^2+e^1} \approx \mathbf{0.09}$$ $$\pi_\theta = [0.67, 0.24, 0.09]$$ 

## Step 3: Compute the Elements of the KL Formula
The formula is: $\sum \pi_{\text{teacher}}(w) \cdot \log \left( \frac{\pi_{\text{teacher}}(w)}{\pi_\theta(w)} \right)$
Let's calculate the value inside the sum for each token:

   1. For "blue":
   $$0.98 \cdot \log\left(\frac{0.98}{0.67}\right) = 0.98 \cdot \log(1.46) = 0.98 \cdot 0.378 = \mathbf{0.370}$$ 
   2. For "cloudy":
   $$0.02 \cdot \log\left(\frac{0.02}{0.24}\right) = 0.02 \cdot \log(0.083) = 0.02 \cdot (-2.48) = \mathbf{-0.049}$$ 
   3. For "banana":
   $$0.00 \cdot \log\left(\frac{0.00}{0.09}\right) = \mathbf{0.000}$$ 

## Step 4: Sum Them Up
$$\mathbb{D}_{\text{KL}} = 0.370 + (-0.049) + 0.000 = \mathbf{0.321}$$ 

This scalar value 0.321 tells us how mathematically "far apart" the student's prediction structure is from the teacher. The training algorithm will calculate gradients to drive this number down toward 0.

---

## (2) The Reverse KL Loss (Mode-Seeking vs. Mean-Seeking)
The Thinking Machines paper highlights a critical design choice: using Reverse KL ($\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{teacher}})$) instead of the traditional Forward KL ($\mathbb{D}_{\text{KL}}(\pi_{\text{teacher}} \parallel \pi_\theta)$).
$$\text{Reverse KL} = \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{teacher}}) = \mathbb{E}_{x \sim \pi_\theta} \left[ \log \pi_\theta(x_{t+1} \mid x_{1..t}) - \log \pi_{\text{teacher}}(x_{t+1} \mid x_{1..t}) \right]$$ Fine-tuning with Forward KL forces the student to average its probabilities across all options the teacher likes (mean-seeking). Reverse KL forces the student to choose one specific high-quality strategy and commit to it (mode-seeking).

### Step-by-Step Mathematical Computation
Let's see what happens when a student model takes a "risky" action and generates a bad token that the teacher hates.

* Vocabulary Space: ["reasoning", "hallucination"]
* Teacher Distribution ($\pi_{\text{teacher}}$): [0.999, 0.001] (The teacher heavily penalizes hallucination).
* Student Distribution ($\pi_\theta$): [0.500, 0.500] (The student is confused and unsure).

Let’s compute the penalty for each token choice using the inside of the expectation: $\log \pi_\theta(w) - \log \pi_{\text{teacher}}(w)$.
Scenario A: The Student chooses the correct token "reasoning"
$$\text{Penalty} = \ln(0.500) - \ln(0.999) = -0.693 - (-0.001) = \mathbf{-0.692}$$ 
Scenario B: The Student chooses the bad token "hallucination"
$$\text{Penalty} = \ln(0.500) - \ln(0.001) = -0.693 - (-6.907) = \mathbf{+6.214}$$ 

* The Intuition: Look at the asymmetry. When the student samples a token where the teacher's probability is near zero, the $\ln(\pi_{\text{teacher}})$ term approaches negative infinity ($-\infty$), causing the overall loss to skyrocket (+6.214). This heavily penalizes the student for generating gibberish or incorrect reasoning steps.

------------------------------
## (3) Dense Supervision via Token-Level Advantages
In standard RL, a model writes a 500-word essay, and a reward model gives a single scalar score at the very end (e.g., +1.0 or 0.0). This is sparse reward.
On-policy distillation turns the negative Reverse KL value at every single token step into a dense advantage signal ($A_t$) inside a policy gradient loop:
$$A_t = -\text{Reverse KL}_t = \log \pi_{\text{teacher}}(x_t \mid s_t) - \log \pi_\theta(x_t \mid s_t)$$ 
The model is updated using an importance-sampling policy gradient loss:
$$\mathcal{L}_{\text{PG}}(\theta) = -\sum_{t=1}^T \frac{\pi_\theta(x_t \mid s_t)}{\pi_{\text{old}}(x_t \mid s_t)} \cdot A_t$$ 

### Step-by-Step Mathematical Computation
Suppose the student model is mid-generation. It has just generated a token.

* The student's old probability for this token was: $\pi_{\text{old}} = 0.40$
* The student's current active probability is: $\pi_\theta = 0.42$
* The teacher's probability for this exact token is: $\pi_{\text{teacher}} = 0.90$

Let's calculate the dense step-by-step update:

   1. Compute the Dense Advantage ($A_t$):
   $$A_t = \ln(0.90) - \ln(0.42) = -0.105 - (-0.867) = \mathbf{+0.762}$$ Because the teacher liked this token much more than the student did, the advantage is positive (+0.762).
   2. Compute the Importance Sampling Ratio ($r_t$):
   $$r_t = \frac{\pi_\theta}{\pi_{\text{old}}} = \frac{0.42}{0.40} = \mathbf{1.05}$$ 
   3. Compute the Final Loss Component for this token:
   $$\mathcal{L} = -(1.05 \times 0.762) = \mathbf{-0.800}$$ 


* Intuition: Minimizing a negative loss means maximizing the probability of this action. This dense per-token tracking allows on-policy distillation to learn 50x to 100x faster than standard sparse RL because the model doesn't have to guess which word caused a final wrong answer—every token receives an immediate, precise grade.

------------------------------
## (4) The Continual Learning Drift Dilemma
A fascinating discovery in the Thinking Machines paper addresses catastrophic forgetting. When you take a model that already knows how to act like an assistant (e.g., instruction-following) and train it on new raw data (like internal medical or company documents), its structural assistant behaviors collapse.
Researchers often try to fix this by taking the model, generating samples from it, and using standard Supervised Fine-Tuning (SFT) on those self-generated samples to preserve its behavior. The paper mathematically proves this fails.

### Step-by-Step Mathematical Computation of the Drift
Why does standard SFT on self-samples fail while On-Policy Distillation succeeds?

1. The Finite Batch Variance Problem: In theory, if a model trains on its own perfect probability distribution, the gradient should be zero. However, computers train on finite batches (e.g., 64 sentences at a time).
2. The SFT Gradient Update:
$$\nabla_\theta \mathcal{L}_{\text{SFT}} = -\frac{1}{B}\sum_{i=1}^B \nabla_\theta \log \pi_\theta(x_i)$$ Because the batch is finite, the empirical distribution of the batch will never perfectly match the exact mathematical policy $\pi_\theta$ . This introduces gradient noise. The model updates its parameters slightly based on this noise: $\theta_{\text{new}} = \theta + \Delta \theta$.
3. The Compounding Shift: At the next training step, the model's distribution has shifted to $\pi_{\theta_{\text{new}}}$. The next batch is sampled from this new, slightly degraded distribution. Over thousands of steps, this transforms the setup into an off-policy training loop, resulting in catastrophic drift.
4. How On-Policy Distillation Fixes It: By holding a frozen copy of the original model as the Teacher, the target distribution $\pi_{\text{teacher}}$ never moves. Even if a finite batch introduces noise, the Reverse KL term forces the student right back to the stable anchor.


## (5) Knowledge Distillation as Functional Regularization
At a PhD level, Knowledge Distillation is framed as a optimization problem mapping a complex continuous function $f_{\text{teacher}}: \mathcal{X} \to \mathcal{Y}$ to a lower-capacity hypothesis class $\mathcal{F}_{\text{student}}$ parameterized by $\theta$.
Instead of minimizing empirical risk over an unaligned dataset, KD optimizes the student within a bounded information-theoretic constraint:
$$\min_{\theta} \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathcal{L}_{\text{task}}(f_\theta(x), y) + \lambda \cdot \mathbb{D}_{\text{f}}(P_{\text{teacher}}(Y \mid x) \parallel P_\theta(Y \mid x)) \right]$$ 
Where $\mathbb{D}_{\text{f}}$ is an arbitrary $f$-divergence (such as Total Variation, KL, or Rényi divergence) acting as a smooth structural regularizer. This forces the student's optimization trajectory to favor flatter minima in the loss landscape, yielding superior generalization bounds compared to standard maximum likelihood estimation (MLE).
### 5.1. Policy Distillation in Auto-Regressive Decision Space
To understand Policy Distillation rigorously, an auto-regressive Language Model must be formalized as a deterministic, finite-horizon Markov Decision Process (MDP) denoted by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ where:

* The state space $\mathcal{S}$ is combinatorial: $\mathcal{S} = \mathcal{V}^{\le T}$, representing all valid text sequences up to length $T$.
* The action space $\mathcal{A}$ is the discrete vocabulary $\mathcal{V}$.
* The transition dynamics are deterministic and non-Markovian in terms of tokens, but strictly Markovian over the state space: $\mathcal{P}(s_{t+1} \mid s_t, a_t)$ appends action $a_t$ to state $s_t$ with probability 1.

Policy distillation is the process of behavioral cloning or policy alignment within this state space, optimizing the student policy $\pi_\theta(a \mid s)$ to approximate the teacher policy $\pi_* (a \mid s)$.
### 5.2. LLM Distillation vs. Policy Distillation: The Theoretical Divide
While often conflated in applied engineering, their formal mechanics are distinct:

* LLM Distillation (Data-Level Compression): Operates primarily in the data space. It solves a conditional density estimation problem over a static corpus $\mathcal{D}$. It minimizes a standard token-level empirical risk:
$$\mathcal{L}_{\text{LLM}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]$$ This treats the data generation process as independent of the student's parameter updates.
* Policy Distillation (Distributional Alignment): Invokes the policy space. It views generation as a sequential trajectory rollout where actions influence future states. It accounts for the state-visitation distribution $\rho_\pi(s)$ induced by the active policy, forcing the alignment of decision boundaries across multi-turn horizons rather than static token sequences.

## (6) Off-Policy Distillation and the Exposure Bias Objective
In Off-Policy Distillation (e.g., standard Supervised Fine-Tuning on teacher-generated datasets), the optimization objective is defined over the state-visitation distribution induced by the teacher policy $\rho_{\pi_{\text{teacher}}}(s)$:
$$\mathcal{L}_{\text{off-policy}}(\theta) = \mathbb{E}_{s \sim \rho_{\pi_{\text{teacher}}}} \left[ \mathbb{D}_{\text{KL}}\left(\pi_{\text{teacher}}(\cdot \mid s) \parallel \pi_\theta(\cdot \mid s)\right) \right]$$ 
## The Compounding Error Proof (The Curse of Off-Policy)
Because the student optimizes purely on states sampled from $\rho_{\pi_{\text{teacher}}}$, it never experiences states sampled from its own flawed distribution $\rho_{\pi_\theta}$.
Let the student policy error bounded at any given state be $\epsilon$, such that $\max_s \mathbb{D}_{\text{TV}}(\pi_{\text{teacher}}(\cdot \mid s) \parallel \pi_\theta(\cdot \mid s)) \le \epsilon$. During auto-regressive decoding (inference), the student drifts off the teacher's state-space trajectory. By Ross, Gordon, and Bagnell (DAGGER framework), the total variation distance between the teacher's trajectory distribution and the student's trajectory distribution compounds quadratically over a horizon $T$:
$$\mathbb{D}_{\text{TV}}(\rho_{\pi_{\text{teacher}}}(\tau) \parallel \rho_{\pi_\theta}(\tau)) \le \mathcal{O}(\epsilon T^2)$$ 
This $\mathcal{O}(T^2)$ compounding error is the mathematical proof of exposure bias. The student encounters out-of-distribution states, leading to catastrophic degradation or hallucination during long-text generation.

## (7) On-Policy Distillation and Minimax Optimization
To eliminate the $\mathcal{O}(T^2)$ compounding error, On-Policy Distillation shifts the expectation of the objective function to the student's own state-visitation distribution $\rho_{\pi_\theta}(s)$:
$$\mathcal{L}_{\text{on-policy}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^{T} \mathbb{D}_{\text{KL}}\left(\pi_{\text{teacher}}(\cdot \mid s_t) \parallel \pi_\theta(\cdot \mid s_t)\right) \right]$$ 
By sampling trajectories $\tau = (s_1, a_1, \dots, s_T)$ directly from $\pi_\theta$, the student explores the exact regions of the state space where its current parameters are prone to error.

## Modern Paradigm: MiniLLM and Forward KL Formulations
In standard RL, optimizing this requires Policy Gradient methods (like PPO) because the environment distribution depends on $\theta$. However, in On-Policy Distillation, because text generation transitions are deterministic ($s_{t+1} = s_t \cup \{a_t\}$), we can derive the gradient directly.
As proved in contemporary works like MiniLLM (ICLR 2024) [1], optimizing the Forward KL on the student's distribution prevents the student from overestimating low-probability regions of the teacher (which causes the model to generate gibberish). The gradient can be formulated without high-variance reward-based policy gradients by leveraging the exact token probabilities of the teacher evaluated over the student's rollouts. This reduces the trajectory error bound from quadratic to linear:
$$\mathbb{D}_{\text{TV}}(\rho_{\pi_{\text{teacher}}}(\tau) \parallel \rho_{\pi_\theta}(\tau)) \le \mathcal{O}(\epsilon T)$$ 
