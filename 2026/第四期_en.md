[English](./第四期_en.md) | [中文](./第四期.md)

# ArXiv Weekly - January 2026 Week 4

> **Keywords of the Week**: Post-LayerNorm Renaissance, Inference Neuron Intervention, Generative Novel View Synthesis, On-Policy Continual Learning, Critical Sharpness Monitoring

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Generational Shift in Deep Learning Infrastructure](#2-generational-shift-in-deep-learning-infrastructure)
  - [2.1 Post-LayerNorm Is Back](#21-post-layernorm-is-back)
- [3. Training Dynamics and Optimization Theory](#3-training-dynamics-and-optimization-theory)
  - [3.1 A Scalable Measure of Loss Landscape Curvature](#31-a-scalable-measure-of-loss-landscape-curvature)
- [4. Mechanistic Interpretability and Inference Control](#4-mechanistic-interpretability-and-inference-control)
  - [4.1 AdaRAS: Identifying and Transferring Reasoning-Critical Neurons](#41-adaras-identifying-and-transferring-reasoning-critical-neurons)
- [5. Generative Turn in Computer Vision](#5-generative-turn-in-computer-vision)
  - [5.1 AnyView: Synthesizing Any Novel View](#51-anyview-synthesizing-any-novel-view)
- [6. Continual Learning and Self-Evolution](#6-continual-learning-and-self-evolution)
  - [6.1 Self-Distillation Enables Continual Learning](#61-self-distillation-enables-continual-learning)
- [8. Summary and Outlook](#8-summary-and-outlook)

---

## **1. Abstract**

From January 23 to January 29, 2026, the global computer science community, especially the field of artificial intelligence, is experiencing a severe paradigm shock. This shock does not stem from the release of a single model, but from the entire community's profound reflection on the diminishing marginal utility of "Brute-force Scaling," and a collective return to more efficient, controllable, and interpretable architectures.

In the background noise of this week, the influence of DeepSeek V4 and its predecessor R1 lingers. The "Engram" architecture disclosed by the DeepSeek team in mid-January via arXiv:2601.07372 introduced an O(1) complexity conditional memory mechanism, effectively declaring the arrival of the era of separation between "memory" and "computation" in Transformer architectures. The subsequent FlashMLA code leak (January 21) and NVIDIA's single-day market value evaporation of $60 billion on January 27 have become grand footnotes to this week's academic publications. These market and technological events jointly point to a core proposition: if simply increasing parameter scale is no longer the only or optimal path to AGI, where should underlying architectural innovation go?

It is in this anxiety and exploration of the "Post-Scaling Law" era that the arXiv CS section has seen a batch of weighty research results this week. We observe that the focus of research is significantly shifting from "how to train larger models" to "how to train models smarter" and "how to precisely control models."

This week's literature presents three distinct technical undercurrents:

1.  **Retro and Evolution of Infrastructure**: Research represented by **Keel** challenges the Pre-LayerNorm paradigm that has dominated the industry for years, revitalizing Post-LayerNorm by introducing Highway mechanisms, attempting to conquer the training stability problems of ultra-deep networks with 1000+ layers.
2.  **Transparency and Intervention of Inference Process**: Works like **AdaRAS** are no longer satisfied with text-level explanations of Chain-of-Thought (CoT), but go deep into the neuron level, performing "neurosurgical" real-time intervention on the model's reasoning process through Activation Steering.
3.  **Cognitization of Visual Generation**: In the field of computer vision, **AnyView** marks the complete crossover of Novel View Synthesis (NVS) from geometric reconstruction to generative cognition, where models begin to possess the ability of "reasonable hallucination" in dynamic blind spots.

Based on new arXiv papers released from January 23 to 29, 2026, this report selects 6 milestone literatures for in-depth analysis. We not only analyze their technological innovations and experimental results but also deconstruct the mathematical intuition behind them from the principle level, and explore the far-reaching impact of these technologies in the current turbulent technological landscape.

---

## **2. Generational Shift in Deep Learning Infrastructure**

In the past five years, Pre-LayerNorm has almost become the standard configuration for Large Language Models (LLMs). From GPT-3 to Llama 3, the industry generally adopts Pre-LN because it has excellent stability in the early stage of training and does not require complex Warm-up strategies to converge. However, this stability comes at a price—theoretical analysis shows that Pre-LN dilutes the gradient contribution of deep networks, limiting the final expressive power of the model.

**"Post-LayerNorm Is Back: Stable, ExpressivE, and Deep"** published this week may be one of the most important papers in the field of infrastructure this year. DeepSeek's success proved the importance of underlying operator optimization, while the Keel architecture proves that topological structure optimization has equally huge potential.

### **2.1 Post-LayerNorm Is Back**

> **Post-LayerNorm Is Back: Stable, ExpressivE, and Deep**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.19895-b31b1b.svg)](https://arxiv.org/abs/2601.19895) [**PDF Download**](../Resource/第四期/2601.19895v1.pdf)

<div align="center">
  <img src="../Resource/第四期/figures/2601.19895v1_Figure2.png" width="80%">
  <br>
  <em>Figure 1: Keel architecture and Post-LN improvement diagram (Source: Original Paper Figure 2)</em>
</div>

#### **2.1.1 Core Pain Point: The Dilemma of Pre-LN vs. Post-LN**

To understand Keel's innovation, we must first review the battle of normalization positions in the Transformer architecture.

*   **Pre-LayerNorm (Pre-LN)**: Structure is $x_{l+1} = x_l + F(\text{LN}(x_l))$.
    In this structure, the Residual Connection is dominant, and the non-linear transformation $F$ is located after normalization. Mathematical derivation shows that as the number of layers $L$ increases, the variance of the output layer grows linearly, causing the gradient norm of the bottom parameters to be roughly proportional to $1/\sqrt{L}$ during backpropagation. This means that in extremely deep networks, the weights of the bottom layers can hardly be effectively updated, and the model actually "degenerates" into an ensemble of shallow networks.
*   **Post-LayerNorm (Post-LN)**: Structure is $x_{l+1} = \text{LN}(x_l + F(x_l))$.
    This is the structure adopted by the original Transformer paper (Attention is All You Need). Its advantage is that the input of each layer is normalized, maintaining the consistency of data distribution, and theoretically possessing stronger function approximation capabilities. However, its fatal weakness is gradient explosion. During backpropagation, gradients may be amplified when passing through each $\text{LN}$ layer, leading to extremely unstable training, usually requiring extremely careful design of learning rate Warm-up and parameter initialization strategies.

#### **2.1.2 Core Breakthrough: Keel Architecture and Highway Gating Mechanism**

Paper authors Chen Chen and Lai Wei propose that the instability of Post-LN essentially stems from the uncontrollable signal mixing introduced by ResNet-style direct addition ($x + F(x)$) in deep networks. To solve this problem, the Keel architecture introduces the design philosophy of **Highway Networks**.

Keel replaces the traditional residual connection with a Gated Connection. Its core formula can be formally expressed as:

$$x_{l+1} = \text{LN}(\alpha x_l + (1-\alpha) F(x_l))$$

Or a more complex dynamic gating form:

$$x_{l+1} = \text{LN}(g_l \odot x_l + (1-g_l) \odot F(x_l))$$
$$g_l = \sigma(W_g x_l + b_g)$$

Where $\alpha$ or $g_l$ is the gating coefficient.

**Mechanism Analysis**:

1.  **Dynamic Signal Balance**: The gating mechanism allows the model to autonomously learn how much "original information" to retain and how much "transformed information" to incorporate. In the early stage of training, gating often tends to retain original information (i.e., $\alpha$ is close to 1), which allows gradients to flow losslessly to the bottom layers like in Pre-LN, ensuring startup stability.
2.  **Release of Expressive Power**: As training progresses, the gate gradually opens, allowing $F(x_l)$ to contribute more non-linear transformations. Since the normalization operation $\text{LN}$ is at the end, Keel maintains the high expressiveness of Post-LN, enabling each layer of the deep network to have a substantial impact on the final output.

#### **2.1.3 Experimental Verification: Challenging the Limit of 1000 Layers**

To verify Keel's limit capabilities, the research team conducted comparative experiments under extremely harsh conditions.

**Training Stability and Convergence Curve**

When the learning rate is set to an aggressive $\eta$:
*   **Standard Post-LN**: Gradient explosion occurs in the early stage of training, Loss diverges, and fails to converge.
*   **Pre-LN**: Although it can converge, the Loss decline curve is flat, and oscillation occurs in deep settings.
*   **Keel**: Shows an extremely smooth convergence curve, and the convergence speed is significantly faster than Pre-LN. This indicates that Highway Gating effectively smooths the Loss Landscape.

**Depth Scaling**

This is the most shocking experimental part of this paper. Researchers built a Transformer model with a depth of over **1000 layers**.
*   **Pre-LN**: As the number of layers increases, performance improvement quickly saturates. This is because gradient decay causes the newly added layers to be actually in an "idling" state.
*   **Keel**: Performance shows a continuous upward trend with increasing depth. At 1000 layers, its Perplexity is significantly better than the Pre-LN model with the same amount of parameters.

**Comprehensive Capability Assessment**

In a comprehensive Benchmark containing multilingual understanding, math code, common sense reasoning, and other tasks, Keel achieved a total victory:

| Capability | Pre-LN Average (%) | Keel Average (%) | Relative Gain |
| :---- | :---- | :---- | :---- |
| **Multilingual** | 66.4 | **70.8** | **+6.6%** |
| **General Knowledge** | 63.6 | **66.4** | **+4.4%** |
| **Math & Code** | 38.6 | **45.0** | **+16.5%** |
| **Overall Average** | 58.7 | **62.5** | **+6.5%** |

**Deep Insight**:

Keel's huge improvement (+16.5%) on "Math & Code" tasks is particularly noteworthy. Such tasks usually require extremely long logical reasoning chains. By supporting deeper network structures, Keel actually increases the **Sequential Computation Capacity** of the model. If MoE (like DeepSeek V4) increases memory capacity by increasing width, then Keel increases reasoning depth by increasing depth. Future top models are very likely to adopt a hybrid architecture of **MoE (Width/Memory) + Keel (Depth/Reasoning)**.

---

## **3. Training Dynamics and Optimization Theory**

As model scale grows, alchemists are increasingly like driving a spaceship without instruments. We know the model is converging, but we don't know if it is on the "Edge of Stability." Traditional optimization theory indicators (such as eigenvalues of the Hessian matrix) are computationally unacceptable for 70B+ parameter models.

This week, a research team from Meta Superintelligence Labs published **"A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs"**, filling this theoretical vacuum.

### **3.1 A Scalable Measure of Loss Landscape Curvature**

> **A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.16979-b31b1b.svg)](https://arxiv.org/abs/2601.16979) [**PDF Download**](../Resource/第四期/2601.16979v1.pdf)

<div align="center">
  <img src="../Resource/第四期/figures/2601.16979v1_p3_0.png" width="80%">
  <br>
  <em>Figure 2: Critical Sharpness monitoring diagram</em>
</div>

#### **3.1.1 Theoretical Background: Hessian Matrix and Sharpness**

In optimization theory, the **Sharpness** of the loss function surface is determined by the maximum eigenvalue $\lambda_{\max}$ of the Hessian Matrix (second-order derivative matrix).

*   **Flat Minima**: $\lambda_{\max}$ is small, meaning that small perturbations of parameters will not cause drastic changes in Loss. This is usually associated with good **generalization ability**.
*   **Sharp Minima**: $\lambda_{\max}$ is large, meaning the model is on the edge of a "cliff," and generalization ability is usually poor.

However, calculating the Hessian matrix eigenvalues of a 7B parameter model requires huge computational resources (usually requiring Hessian-vector products), making real-time monitoring during pre-training almost impossible.

#### **3.1.2 Innovative Method: Critical Sharpness**

To solve the scalability problem, the paper proposes a new metric—**Critical Sharpness**. This metric no longer attempts to calculate all eigenvalues of the Hessian matrix precisely but uses the first moment estimation of the gradient to approximate the local curvature.

Its core idea is to measure how large a Step Size will cause Loss divergence in the current gradient direction. The reciprocal of this "critical step size" is the Critical Sharpness. This method only requires standard forward and backward propagation, with almost no extra computational overhead.

#### **3.1.3 Experimental Findings: The "Edge of Stability" Effect of LLMs**

Using this metric, the research team observed the **"Edge of Stability" (EoS)** phenomenon on a 7B parameter scale LLM for the first time.

*   **Phenomenon Description**: During training, the sharpness of the model gradually rises (i.e., the Loss surface becomes steeper) until it reaches a critical threshold. Thereafter, sharpness oscillates around this threshold.
*   **Learning Rate Interaction**: Experiments show that the choice of Learning Rate directly determines the upper limit of this sharpness. A high learning rate forces the model to stay in a flatter region (because it diverges once it enters a sharp region), thereby indirectly improving generalization ability.
*   **Fine-tuning Guidance**: The paper also proposes the concept of "Relative Critical Sharpness" to guide the data mixing ratio during Fine-tuning. Experiments found that when the mixing ratio is 0.6-0.7 DCLM (Data Curriculum Learning Mix), the model achieves the best balance between maintaining original task capabilities (Basin Retention) and learning new tasks.

**Deep Insight**:

This research provides a real dashboard for "alchemy." In the past, our adjustment of learning rate and Batch Size often relied on empirical formulas (such as Scaling Laws); now we can dynamically adjust hyperparameters by monitoring "Critical Sharpness" in real-time. For example, when sharpness is too low, Batch Size can be increased to accelerate convergence; when sharpness touches the critical value, learning rate decay should be turned on. This has extremely high economic value for reducing the cost of large model training (often millions of dollars).

---

## **4. Mechanistic Interpretability and Inference Control**

If Keel and Critical Sharpness are optimizing the "body" of the model, then **AdaRAS** is attempting to control the "mind" of the model. Today, as Reasoning Models become increasingly powerful, simply knowing what the model outputs is far from enough; we need to know *how* it reasons and be able to intervene before it makes a mistake.

**"Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering"** published this week provides us with a precision scalpel.

### **4.1 AdaRAS: Identifying and Transferring Reasoning-Critical Neurons**

> **Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.19847-b31b1b.svg)](https://arxiv.org/abs/2601.19847) [**PDF Download**](../Resource/第四期/2601.19847v1.pdf)

<div align="center">
  <img src="../Resource/第四期/figures/2601.19847v1_Figure1.png" width="80%">
  <br>
  <em>Figure 3: AdaRAS activation steering mechanism (Source: Original Paper Figure 1)</em>
</div>

#### **4.1.1 Problem Background: Opacity of CoT and Limitations of Self-Correction**

Although Chain-of-Thought (CoT) allows models to output reasoning steps, research shows that the CoT output by models is sometimes "Post-hoc Rationalization" and does not completely represent their internal true reasoning path. In addition, research by the DeepSeek team in the same period points out that when models commit deep logical errors, they often cannot correct them through internal self-reflection (i.e., the "Self-Correction Paradox").

#### **4.1.2 Technical Core: AdaRAS (Adaptive Reasoning Activation Steering)**

The core of AdaRAS lies in **Activation Steering**. This is a technique derived from Mechanistic Interpretability, aimed at changing the behavior of the model by adding specific vectors (Steering Vectors) to the activation values of neurons.

**Finding "Reasoning Neurons" (RCNs)**

AdaRAS first proposes a **Polarity-aware Mean-difference Criterion**.

*   Researchers collected two activation states of the model on math problems: "reasoning correct" and "reasoning incorrect."
*   Through comparative analysis, neurons with the largest activation difference on correct and incorrect paths and consistent polarity were screened out. These neurons are defined as **Reasoning-Critical Neurons (RCNs)**. Surprisingly, these neurons account for a tiny fraction of the total model parameters but dominate the direction of logical judgment.

**Adaptive Intervention Mechanism**

In the inference stage (Test-time), the AdaRAS system monitors the activation values of RCNs in real-time.

*   **Enhance Correct Signals**: If the activation pattern of RCNs is detected to be biased towards the "correct" distribution, the system applies a positive steering vector to reinforce this trend.
*   **Suppress Wrong Signals**: If the activation pattern is detected to drift, the system applies a reverse vector to correct it.
*   This intervention is **Adaptive**, meaning it dynamically adjusts the intervention intensity according to the uncertainty of the current Token, avoiding interference with simple steps that the model is already confident about.

#### **4.1.3 Experimental Results: AIME Competition-Level Breakthrough**

Experiments were conducted on a Benchmark containing 10 math and code datasets, and the results were impressive.

*   **AIME 2024/2025**: These are real questions from the American Invitational Mathematics Examination, which are extremely difficult. AdaRAS achieved an accuracy improvement of over **13%** on these two datasets respectively.
*   **Training-free**: Most critically, this improvement was achieved **without any parameter updates (Training-free)**. It simply fine-tuned the activation values during the inference stage.
*   **Transferability**: Research found that RCN features identified on one model (such as Llama-3-8B) can be transferred to models of other scales (such as Llama-3-70B) after simple mapping, implying that there may be general "logic reasoning circuits" in neural networks.

**Deep Insight**:

AdaRAS actually opens the precedent for **"Interventionist AI"**. It proves that we can treat large models as an adjustable physical system, forcing the system's trajectory towards our desired state (Truthfulness) through external signals. This is crucial for AI safety—instead of expecting the model to "learn" not to lie, it is better to physically block this behavior by inhibiting relevant neurons the moment it has the thought of lying.

---

## **5. Generative Turn in Computer Vision**

Computer Vision (CV) has long been dominated by the "Geometric Reconstruction" faction (such as Photogrammetry, NeRF, 3D Gaussian Splatting). However, **"AnyView: Synthesizing Any Novel View in Dynamic Scenes"** released this week completely broke this tradition, demonstrating how generative models solve problems that geometric methods cannot solve through "reasonable hallucination."

### **5.1 AnyView: Synthesizing Any Novel View**

> **AnyView: Synthesizing Any Novel View in Dynamic Scenes**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.16982-b31b1b.svg)](https://arxiv.org/abs/2601.16982) [**PDF Download**](../Resource/第四期/2601.16982v1.pdf)

<div align="center">
  <img src="../Resource/第四期/figures/2601.16982v1_Figure1.png" width="80%">
  <br>
  <em>Figure 4: AnyView novel view synthesis effect display (Source: Original Paper Figure 1)</em>
</div>

#### **5.1.1 Pain Point: Dynamic Blind Spots and Geometric Absence**

In Novel View Synthesis (NVS) of dynamic scenes, the biggest challenge is **Occlusion and Large Viewpoint Changes**.

For example, a video captures a car driving by from the left. If we want to generate a video of the car from the right, traditional geometric methods (like NeRF) will fail completely because the original video contains no pixel information of the right side of the car, and geometric reconstruction cannot "create something from nothing."

#### **5.1.2 Solution: Generative Priors and Spatiotemporal Consistency**

<div align="center">
  <img src="../Resource/第四期/figures/2601.16982v1_Figure2.png" width="80%">
  <br>
  <em>Figure 5: AnyView architecture flowchart: Multimodal (pixel+camera) denoising generation based on Diffusion Transformer (Source: Original Paper Figure 2)</em>
</div>

AnyView no longer tries to "reconstruct" the scene, but tries to "generate" the scene. It uses Video Diffusion Models pre-trained on massive video data as priors.

1.  **Implicit Semantic Completion**: When the camera angle turns to a blind spot, AnyView uses its learned world knowledge (e.g., cars are symmetrical, people have two legs) to generate the missing parts. This generation is not random but constrained by the semantics of the visible parts.
2.  **Generalist Spatiotemporal Implicit Representation**: AnyView proposes a Generalist Spatiotemporal Implicit Representation. It can simultaneously encode static 3D geometric information and dynamic 4D motion information.
3.  **Spatiotemporal Attention**: To prevent generated content from flickering in time, the model introduces a cross-frame attention mechanism to ensure that the generated "hallucination" is coherent on the time axis.

#### **5.1.3 Experimental Verification: AnyViewBench**

For fair evaluation, researchers built **AnyViewBench**, containing a large number of dynamic scenes with extreme viewing angles.
Experiments show that AnyView comprehensively crushes traditional NeRF and 3DGS variants on metrics like PSNR and LPIPS. More importantly, in human subjective evaluation (User Study), videos generated by AnyView were considered to have extremely high realism, even from perspectives completely unseen in the original video.

**Deep Insight**:

AnyView's success marks an important turning point in the CV field: **from "Physics-Centric" to "Cognition-Centric"**. In future VR/AR applications, we may no longer need to scan the entire room to build a 3D model, only a part of it, and AI will "brainstorm" the rest based on common sense. This greatly reduces the threshold for content production, but also poses new ethical challenges for authenticity verification.

---

## **6. Continual Learning and Self-Evolution**

If AI is to become a true assistant, it must be able to continuously learn from user usage, rather than starting training from scratch with every update. This week's **SDFT** research provides new ideas for the "Lifelong Learning" of large models.

### **6.1 SDFT: Self-Distillation Fine-Tuning**

> **SDFT: Self-Distillation Fine-Tuning for Continual Learning**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.19897-b31b1b.svg)](https://arxiv.org/abs/2601.19897) [**PDF Download**](../Resource/第四期/2601.19897v1.pdf)


<div align="center">
  <img src="../Resource/第四期/figures/2601.19897v1_Figure2.png" width="80%">
  <br>
  <em>Figure 6: SDFT continual learning framework diagram (Source: Original Paper Figure 2)</em>
</div>

#### **6.1.1 Pain Point: Catastrophic Forgetting**

In Continual Learning (CL) scenarios, models need to continuously learn new tasks (Task A -> Task B -> Task C).
However, neural networks have a fatal weakness: **learning new knowledge leads to forgetting old knowledge**.
For example, after fine-tuning on medical data, a large model might forget how to write code. Traditional solutions (such as Replay, Parameter Isolation) often require saving a large amount of old data or increasing model parameters.

#### **6.1.2 Solution: SDFT**

SDFT (Self-Distillation Fine-Tuning) proposes a **No Old Data Required** continual learning fine-tuning method.

1.  **Self-Distillation**:
    *   When learning a new task, use the model state **before fine-tuning (Old Model)** as the Teacher.
    *   The model currently being fine-tuned (New Model) acts as the Student.
    *   Require the Student's output distribution (Logits) to be as close as possible to the Teacher's response to the same input while learning new task data.
    *   This preserves the model's ability to represent general knowledge.
2.  **On-Policy Optimization**:
    *   SDFT adopts an On-Policy approach, distilling directly on the data distribution of the current new task without the need for additional auxiliary datasets.

#### **6.1.3 Core Advantages**

*   **No Replay**: No need to save data from old tasks, good privacy, and low storage cost.
*   **Strong Universality**: Applicable to various fine-tuning scenarios (Full Fine-tuning, LoRA, etc.).
*   **Superior Performance**: Experiments show that SDFT effectively adapts to specific tasks in new fields while retaining general capabilities (such as MMLU scores), significantly alleviating the problem of catastrophic forgetting.

---

## **8. Summary and Outlook**

Looking back at this week's arXiv selected literature, we clearly see several key vectors of AI technology evolution:

1.  **Architecture Level**: Shifting from simple layer stacking (Pre-LN) to refined topological design (Keel/Post-LN), pursuing a balance between depth and width.
2.  **Training Level**: Shifting from blind computing power stacking to refined regulation based on theoretical indicators (Critical Sharpness), pursuing training efficiency and stability.
3.  **Inference Level**: Shifting from black-box output to white-box intervention (AdaRAS), pursuing interpretability and controllability.
4.  **Generation Level**: Shifting from physical geometric reconstruction to cognitive generation (AnyView), pursuing semantic consistency and creativity.
5.  **Evolution Level**: Shifting from static models to dynamic adaptation (SDFT), pursuing lifelong learning capabilities.

This series of changes indicates that the AI field is undergoing a transformation from "barbaric growth" to "intensive cultivation." With the rise of open-source forces like DeepSeek, this deep exploration of underlying principles and the extreme pursuit of efficiency will become the main theme for a long time to come.
