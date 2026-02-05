[English](./第五期_en.md) | [中文](./第五期.md)

# ArXiv Weekly - Feb 2026 Issue 5

> **Keywords of the Week**: Pixel Diffusion, Deep Research Agent, Diagnostic Guardrail, Dynamic RL, Reward-free Alignment, BCI Long Context, Scientific Illustration, Previsualization

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Fundamental Reconstruction of Visual Generation: Breaking the Shackles of Latent Space](#2-fundamental-reconstruction-of-visual-generation-breaking-the-shackles-of-latent-space)
  - [2.1 PixelGen: Pixel Diffusion Beats Latent Diffusion with Perceptual Loss](#21-pixelgen-pixel-diffusion-beats-latent-diffusion-with-perceptual-loss)
- [3. Deep Evolution of Agent Cognition: From Retrieval to Research, From Action to Intent](#3-deep-evolution-of-agent-cognition-from-retrieval-to-research-from-action-to-intent)
  - [3.1 Vision-DeepResearch: Inspiring Multimodal LLMs to Perform Deep Research](#31-vision-deepresearch-inspiring-multimodal-llms-to-perform-deep-research)
  - [3.2 AgentDoG: Diagnostic Guardrail for Agentic Systems](#32-agentdog-diagnostic-guardrail-for-agentic-systems)
- [4. System Dynamics of Optimization and Alignment: Dynamic Loop and Reward-Free Mechanisms](#4-system-dynamics-of-optimization-and-alignment-dynamic-loop-and-reward-free-mechanisms)
  - [4.1 RLAnything: An Ecosystem for Dynamic Reinforcement Learning](#41-rlanything-an-ecosystem-for-dynamic-reinforcement-learning)
  - [4.2 RACO: Reward-free Alignment for Conflicting Objectives](#42-raco-reward-free-alignment-for-conflicting-objectives)
- [5. Vertical Domain Architectural Innovation: Specialized Tools for Science and Creativity](#5-vertical-domain-architectural-innovation-specialized-tools-for-science-and-creativity)
  - [5.1 MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](#51-meg-xl-data-efficient-brain-to-text-via-long-context-pre-training)
  - [5.2 PaperBanana: Automating Academic Illustration for AI Scientists](#52-paperbanana-automating-academic-illustration-for-ai-scientists)
  - [5.3 PrevizWhiz: Generative Previsualization for Film Industry](#53-previzwhiz-generative-previsualization-for-film-industry)
- [6. Summary and Outlook: The Dawn of Systemic Intelligence](#6-summary-and-outlook-the-dawn-of-systemic-intelligence)

---

## **1. Abstract**

The most significant technical trends this week can be summarized as the following three "Returns" and "Evolutions":

1.  **Return from "Latent Space" to "Physical Space"**: In the field of computer vision generation, the dominance of Latent Diffusion Models (LDMs) has shown cracks. The emergence of **PixelGen** challenges the fixed mindset of "compressing images via VAE for computational efficiency," proving that with abundant computing power and algorithm optimization, generating directly in pixel space is not only feasible but can achieve fidelity surpassing the compression paradigm. This marks the shift of Generative AI from pursuing "similarity" to pursuing "physical-level precision."
2.  **Evolution from "Information Retrieval" to "Deep Research"**: In the field of Agents, simple RAG (Retrieval-Augmented Generation) can no longer meet complex task requirements. **Vision-DeepResearch** demonstrates how agents can simulate the research path of human PhD students—from preliminary screening of multi-source heterogeneous data to deep verification of multiple entities and scales, and finally to knowledge synthesis—by introducing cold-start supervision and reinforcement learning. The emergence of this capability means that AI is transforming from a mere "tool user" to a "research partner" capable of independent thinking.
3.  **Evolution from "Static Alignment" to "Dynamic Loop"**: In terms of reinforcement learning and safety alignment, **RLAnything** and **RACO** have broken the static deadlock from the perspectives of system architecture and optimization algorithms, respectively. The former constructs a fully dynamic ecosystem where environment, policy, and reward models evolve together, while the latter solves the Pareto optimization problem of conflicting multi-objectives through gradient clipping technology without an explicit reward model. Meanwhile, **AgentDoG** puts a "diagnostic rein" on increasingly powerful agents, shifting from simple interception to deep understanding of behavioral intent.

This report will dismantle the principles, analyze experiments, and assess the impact of these breakthrough works through the following four core chapters, aiming to provide a forward-looking reference with both depth and breadth for professional researchers and technical decision-makers.

---

## **2. Fundamental Reconstruction of Visual Generation: Breaking the Shackles of Latent Space**

In the past few years, image generation tasks in Generative AI have been almost monopolized by Latent Space-based methods, most notably the Stable Diffusion series. Its core logic is: since modeling directly in high-dimensional pixel space is too computationally expensive, images are first compressed into a low-dimensional latent space via a Variational Autoencoder (VAE), and then the diffusion process is performed in this space. However, this "compress-generate-decompress" process inevitably leads to loss of high-frequency details, artifacts, and loss of text-image alignment precision. **PixelGen** published this week formally challenges this paradigm.

### **2.1 PixelGen: Pixel Diffusion Beats Latent Diffusion with Perceptual Loss**

> **PixelGen: Pixel Diffusion Beats Latent Diffusion with Perceptual Loss**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.02493-b31b1b.svg)](https://arxiv.org/abs/2602.02493) [**PDF Download**](../Resource/第五期/2602.02493.pdf)

**Source**: arXiv:2602.02493 [cs.CV, cs.AI] **Date**: Feb 2, 2026 **Authors**: Zehong Ma, Peking University

#### **2.1.1 Core Theory and Architecture Innovation**

<div align="center">
  <img src="../Resource/第五期/piexlgen-Overview of PixelGen.png" width="80%">
  <br>
  <em>Figure 1: PixelGen Architecture Overview: Direct diffusion in pixel space, no VAE compression (Source: Original Paper Figure 1)</em>
</div>


The core proposition of **PixelGen** is: with the improvement of hardware capabilities and the progress of optimization algorithms, we no longer need to sacrifice the original fidelity of images for computational efficiency. The study proposes an end-to-end pixel-level diffusion framework that completely discards VAE, latent representation, and all auxiliary stages, modeling directly in the raw **Pixel Space**.

However, training diffusion models directly in pixel space faces huge challenges, mainly because the extremely high dimensionality of pixel space makes the model difficult to converge and prone to overfitting local statistical laws while ignoring global semantic structures. To solve this problem, PixelGen introduces a **Hybrid Perceptual Losses** mechanism, which is the key to its ability to "beat" latent diffusion.

*   **LPIPS Loss (Local Texture Supervision)**: The research team introduced Learned Perceptual Image Patch Similarity (LPIPS) as part of the loss function. LPIPS uses a pre-trained convolutional neural network (such as VGG) to extract features, calculating the difference between the generated image and the real image not only in pixel values but also in texture structure in the feature space. This allows the model to keenly capture high-frequency details such as edges and textures when generating in pixel space, avoiding the blurring problem caused by traditional L2 loss.
*   **DINOv2 Feature Loss (Global Semantic Supervision)**: Local texture alone is not enough; the model also needs to understand global semantics like "this is a cat" or "this is a building." PixelGen utilizes the deep features of the self-supervised visual Transformer (DINOv2) as guidance. DINOv2 has learned strong semantic representations in an unsupervised setting. By minimizing the distance between the generated image and the target in the DINO feature space, PixelGen is forced to learn the high-level topology and semantic layout of the image.

This dual supervision mechanism of "LPIPS + DINO" essentially injects the perceptual bias of the human visual system into the pixel generation process, enabling the model to efficiently lock onto the Image Manifold even without VAE compression.

#### **2.1.2 Experimental Analysis: Dialectics of Fidelity and Efficiency**

PixelGen's experimental results are shocking, especially in comparison with current state-of-the-art Latent Diffusion Models (LDM).

<div align="center">
  <img src="../Resource/第五期/pixelgen-This work shows that pixel diffusion with perceptual loss outperforms latent diffusion. .png" width="80%">
  <br>
  <em>Figure 2: PixelGen vs Latent Diffusion performance comparison with perceptual loss (Source: Original Paper Figure 2)</em>
</div>

**Table 1: Performance Comparison of PixelGen vs Mainstream Latent Diffusion Models on ImageNet-256**

| Architecture | Method Type | Training Epochs | Sampling Steps | Guidance (CFG) | FID (Lower is Better) | IS (Higher is Better) | Core Advantage |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **PixelGen (Ours)** | **Pixel Diffusion** | **80** | 50 (Heun) | **No** | **5.11** | **--** | **Extremely Fast Training, No VAE** |
| REPA-XL/2 | Latent Diffusion | 800 | 250 (Euler) | No | 5.90 | -- | Requires 10x Training Time |
| DDT-XL/2 | Latent Diffusion | 400 | 250 (Euler) | No | 6.27 | -- | Inferior Performance |
| JiT Baseline | Pixel Diffusion | 200k steps | -- | No | 23.67 | -- | Traditional Pixel Methods Fail |
| **PixelGen-XL** | **Pixel Diffusion** | **160** | **50** | **Yes** | **1.83** | **High** | **SOTA Level Performance** |

**Deep Interpretation:**

1.  **Subversion of Training Efficiency**: PixelGen achieves an FID of 5.11 with only 80 Epochs, while the comparable latent diffusion model REPA-XL/2 requires 800 Epochs to reach 5.90. This means that after introducing perceptual loss, the convergence speed in pixel space has not slowed down, but has become more efficient due to direct contact with raw signals.
2.  **Strong Performance without Guidance (CFG-free)**: PixelGen's advantage is particularly evident without Classifier-Free Guidance (CFG). CFG is typically used to enhance semantic alignment but sacrifices diversity. PixelGen's excellent performance without CFG indicates that its base model itself possesses extremely strong semantic understanding capabilities, directly attributed to the introduction of DINOv2 feature loss.
3.  **GenEval Benchmark**: In larger-scale text-to-image tasks, PixelGen scored a total of 0.79 on the GenEval benchmark, proving that it is not only suitable for closed-category generation like ImageNet but also capable of competing with top-tier LDMs in open-domain text generation.

#### **2.1.3 Industry Impact and Future Outlook**

PixelGen's success marks an important turning point for Generative AI. It implies that VAE might just be a transitional compromise in the era of limited computing power. In fields with zero tolerance for artifacts, such as scientific visualization, medical image generation (requiring extremely high precision), and film special effects, architectures like PixelGen that "directly connect to the physical world" will become the new mainstream. In the future, we may see more end-to-end generation architectures combining feature supervision from strong discriminative models (such as DINO, CLIP), completely eliminating the historical term "compression artifacts."

---

## **3. Deep Evolution of Agent Cognition: From Retrieval to Research, From Action to Intent**

If the progress of visual generation lies in "fidelity," then the progress in the Agent field lies in "depth." Two core papers this week, **Vision-DeepResearch** and **AgentDoG**, define the standards for the new generation of agents from the upper limit of capability (deep research) and the lower limit (safety diagnosis), respectively.

### **3.1 Vision-DeepResearch: Inspiring Multimodal LLMs to Perform Deep Research**

> **Vision-DeepResearch: Inspiring Multimodal LLMs to Perform Deep Research**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.22060-b31b1b.svg)](https://arxiv.org/abs/2601.22060) [**PDF Download**](../Resource/第五期/2601.22060.pdf)

**Source**: arXiv:2601.22060 [cs.CV, cs.AI] **Date**: Jan 29, 2026 **Authors**: OpenAI45Lab & Collaborators

#### **3.1.1 Core Problem: Limitations of RAG and the Rise of Deep Research**

<div align="center">
  <img src="../Resource/第五期/vision-deepresearch-We identify two key limitations of existing multimodal deep-research paradigms for image search. .png" width="80%">
  <br>
  <em>Figure 3: Two key limitations of existing RAG paradigms in multimodal deep research (Source: Original Paper Figure 1)</em>
</div>


Existing Multimodal Large Language Models (MLLMs) perform reasonably well on immediate, shallow questions, but are often helpless when dealing with complex tasks requiring long-cycle, multi-step investigations like a PhD student. Traditional RAG (Retrieval-Augmented Generation) systems suffer from severe path dependence—if the initial retrieval fails, or the retrieved images contain complex multi-entity information, the model often cannot perform secondary correction. Moreover, existing benchmarks usually assume a single query can solve the problem, ignoring the visual noise and information fog prevalent in the real world.

**Vision-DeepResearch** proposes a brand-new paradigm aiming to endow AI agents with traits similar to human researchers: **Hypothesis Generation, Multi-source Verification, Iterative Correction, Deep Synthesis**.

#### **3.1.2 Key Technical Innovations**

<div align="center">
  <img src="../Resource/第五期/vision-deepresearch-Our Data Pipeline.png" width="80%">
  <br>
  <em>Figure 4: Vision-DeepResearch Data Pipeline and Multi-Scale Search Mechanism (Source: Original Paper Figure 2)</em>
</div>


The study proposes a complex system architecture containing three core components:

1.  **Multi-Entity & Multi-Scale Visual Search**:
    Faced with an image containing hundreds of objects or complex scenes, direct global search often yields extremely noisy results. Vision-DeepResearch introduces a **Dynamic Cropping** mechanism:
    *   The model first understands the image globally via a visual encoder.
    *   Based on reasoning needs, the model automatically locates and crops key Regions of Interest (ROIs), such as logos in corners, landmarks in the background, or blurred text.
    *   These cropped images act as independent "Visual Actions" submitted to the search engine in parallel or series. This multi-scale strategy significantly improves recall in complex visual scenes.
2.  **Cold-Start Supervision**: To teach the model how to "do research," the research team constructed a high-quality dataset containing 30,000 multimodal trajectories. These trajectories are not just "question-answer" pairs, but records of complete thought processes: including how the model decomposes problems, selects search keywords, browses web pages, decides to crop images, and corrects search directions based on feedback. Through Supervised Fine-Tuning (SFT) on this dataset, the model gains initial research strategies.
3.  **Candidates Crossover Algorithm**: In the inference phase, the system does not rely on a single reasoning path. It deploys multiple LLM candidates with different parameter settings to explore the solution space in parallel. Subsequently, a crossover algorithm synthesizes and verifies the evidence found by different candidates. This mechanism is similar to "brainstorming" in a research team, effectively reducing hallucinations produced by a single model.

#### **3.1.3 Experimental Validation: Surpassing Closed-Source Giants**

On the specially designed **VDR-Bench** (containing 100 PhD-level research tasks), Vision-DeepResearch demonstrated dominant performance.

**Table 2: Performance Comparison on DeepResearch Bench**

| Model/System | Core Mechanism | Comprehensive Score | Gap to SOTA |
| :---- | :---- | :---- | :---- |
| **Vision-DeepResearch (Ours)** | **SFT + RL + Multi-Scale Search** | **46.21** | **--** |
| Claude Researcher | Parallel Scaling | ~40.5 | -5.7 |
| Nvidia AIQ Research Assistant | Agent Workflow | ~38.0 | -8.2 |
| Perplexity Research | RAG + Search | ~35.5 | -10.7 |
| GPT-5 Agent Workflow | Generalist LLM | ~42.0 | -4.2 |

Experimental results show that although general models like GPT-5 have strong basic capabilities, their ability to solve long-chain complex problems is still inferior to the specifically optimized Vision-DeepResearch in the absence of specialized "research methodology" training. Especially in tasks requiring evidence chain splicing across multiple images and long texts, Vision-DeepResearch's multi-scale search strategy played a decisive role.

### **3.2 AgentDoG: Diagnostic Guardrail for Agentic Systems**

> **AgentDoG: Diagnostic Guardrail for Agentic Systems**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.18491-b31b1b.svg)](https://arxiv.org/abs/2601.18491) [**PDF Download**](../Resource/第五期/2601.18491.pdf)

**Source**: arXiv:2601.18491 **Date**: Jan 29, 2026 **Authors**: Dongrui Liu, Shanghai AI Laboratory

#### **3.2.1 Core Problem: Failure of Black-Box Guardrails**

As agents increasingly access file systems, network interfaces, and code execution environments, their security risks rise exponentially. Existing security guardrails are usually based on rules or simple classifiers, only giving binary "intercept/pass" decisions. This mechanism has two fatal flaws:

1.  **Lack of Context Awareness**: File deletion is safe in a "clear cache" task, but may be malicious in a "data analysis" task. Traditional guardrails struggle to distinguish this nuance.
2.  **Lack of Explainability**: When an agent is intercepted, developers and users often don't know why, leading to debugging difficulties, and potentially reducing usability rather than enhancing security.

#### **3.2.2 AgentDoG Framework Details**

<div align="center">
  <img src="../Resource/第五期/agentdog-Overview of the three orthogonal dimensions of the agentic safety taxonomy.png" width="80%">
  <br>
  <em>Figure 5: AgentDoG 3D Risk Taxonomy Overview (Source: Original Paper Figure 1)</em>
</div>


The core contribution of **AgentDoG** (Diagnostic Guardrail) is that it is not just an interceptor, but a "pathology diagnostician." It is built on a brand-new **3D Risk Taxonomy**:

*   **Dimension 1: Source (Where)**: Is the risk from malicious user instructions (User Malicious), misleading environmental feedback (Env Noise), or the model's own hallucination (Model Hallucination)?
*   **Dimension 2: Failure Mode (How)**: Is it procedural deviation (failure to follow SOP), unauthorized action (Unauthorized Action), or compliance with adversarial attacks (Compliance)?
*   **Dimension 3: Consequence (What)**: Does it lead to privacy leakage, system integrity destruction, or resource abuse?

<div align="center">
  <img src="../Resource/第五期/agentdog-Three-stage, planner-based pipeline for multi-step agent safety trajectory synthesis.png" width="80%">
  <br>
  <em>Figure 6: AgentDoG Planner-based Multi-step Safety Trajectory Synthesis Pipeline (Source: Original Paper Figure 2)</em>
</div>


Based on this taxonomy, the research team generated a large-scale synthetic dataset **ATBench**, containing 100,000 detailed annotated agent interaction trajectories. The AgentDoG model (available in 4B, 7B, 8B versions) was trained on this data, learning not only to identify risks but also to generate structured **Diagnostic Reports**.

#### **3.2.3 Experiments and Applications**

Experiments show that AgentDoG reached SOTA levels in risk identification accuracy, and more importantly, its generated diagnostic reports were rated as extremely valuable in human evaluations. For example, when an agent attempts to access a restricted API, AgentDoG not only intercepts but also outputs: "Interception Reason: Unauthorized Access. Root Cause Diagnosis: Agent failed to call authentication interface before attempting to fetch sensitive data (Violation of SOP). Suggested Fix: Add authentication step in planning layer."

This **Explainable Safety** is a prerequisite for agents to move towards large-scale commercial application. The emergence of AgentDoG fills the huge gap between agent "capability" and "control."

---

## **4. System Dynamics of Optimization and Alignment: Dynamic Loop and Reward-Free Mechanisms**

In the underlying training and alignment mechanisms of agents, research this week shows how to break through algorithmic limitations through system design optimization.

### **4.1 RLAnything: An Ecosystem for Dynamic Reinforcement Learning**

> **RLAnything: An Ecosystem for Dynamic Reinforcement Learning**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.02488-b31b1b.svg)](https://arxiv.org/abs/2602.02488) [**PDF Download**](../Resource/第五期/2602.02488.pdf)

**Source**: arXiv:2602.02488 [cs.LG, cs.AI] **Date**: Feb 2, 2026 **Authors**: Yinjie Wang et al.

#### **4.1.1 Pain Points of Static RL Systems**

Traditional Reinforcement Learning (RL) processes usually involve a fixed Environment, a fixed Reward Function, and a Policy to be optimized. However, in open-world tasks, this static setting quickly hits a bottleneck:

*   **Environment Stagnation**: Once the agent masters the training environment, it stops evolving, leading to poor generalization.
*   **Reward Model Collapse**: As the policy improves, the behavior distribution generated by the agent deviates from the training distribution of the reward model, leading the reward model to give erroneous high scores (Goodhart's Law / Reward Hacking).

#### **4.1.2 Dynamic Trinity Architecture**

<div align="center">
  <img src="../Resource/第五期/RLanything-Motivation and takeaways of our RLAnything framework.png" width="80%">
  <br>
  <em>Figure 7: RLAnything Framework Motivation and Core Advantages (Source: Original Paper Figure 1)</em>
</div>


**RLAnything** proposes a fully dynamic system architecture, treating the environment, policy, and reward model as three coupled, co-evolving dynamic components:

1.  **Policy**: The agent itself, constantly learning to solve harder tasks.
2.  **Reward Model**: Self-updates using data generated by the policy during learning. The system introduces **Consistency Feedback**, utilizing the policy's different performance on similar tasks to calibrate the reward model, enabling it to provide finer-grained Step-wise signals rather than just final success/failure signals.
3.  **Environment**: This is the most innovative point. The system automatically generates new tasks slightly above current capabilities based on the agent's current competency boundary (Auto-Curriculum). This is achieved through an LLM-based "Environment Generator" that analyzes the agent's failure cases and constructs targeted training scenarios.

#### **4.1.3 Performance Leap**

<div align="center">
  <img src="../Resource/第五期/RLAnything-Examples of environment task adaptation based on critic feedback across computer use agent, text-game agent, and coding LLM in our experiments.png" width="80%">
  <br>
  <em>Figure 8: RLAnything Task Adaptation Examples in Different Environments (Source: Original Paper Figure 2)</em>
</div>


The "flywheel effect" generated by this closed-loop system is astonishing. In the **OSWorld** (OS control) benchmark, the Qwen3-VL-8B model equipped with RLAnything improved performance by 9.1% without any human intervention. In **AlfWorld** (text adventure games), accuracy increased by 18.7%. This proves that future RL system competition will no longer be just about algorithms, but about ecosystem evolution efficiency.

### **4.2 RACO: Reward-free Alignment for Conflicting Objectives**

> **RACO: Reward-free Alignment for Conflicting Objectives**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.02495-b31b1b.svg)](https://arxiv.org/abs/2602.02495) [**PDF Download**](../Resource/第五期/2602.02495.pdf)

**Source**: arXiv:2602.02495 [cs.CL, cs.AI] **Date**: Feb 3, 2026 **Authors**: Peter Chen et al.

#### **4.2.1 Dilemma of Multi-Objective Alignment**

When aligning AI to human values, we often face multiple conflicting objectives. The most typical is the conflict between "Helpfulness" and "Harmlessness." Excessive pursuit of harmlessness leads the model to refuse to answer any slightly sensitive questions ("Over-defensiveness"), while excessive pursuit of helpfulness may lead to unsafe content output. Traditional Weighted Sum methods often fail to precisely control this trade-off and rely on unstable Reward Models.

#### **4.2.2 Clipped Conflict-Averse Gradient Descent**

**RACO** (Reward-free Alignment for Conflicting Objectives) proposes a solution without an explicit reward model. Its core is an improvement to the optimization algorithm:

*   **CAGrad (Conflict-Averse Gradient descent)**: A multi-objective optimization algorithm designed to find an update direction that reduces losses for all objectives simultaneously. However, original CAGrad can be overly biased towards an easily optimized objective in some cases.
*   **Clipping Mechanism**: RACO introduces a clipping operation on top of CAGrad, forcing the gradient update direction to fall within a certain neighborhood of the user-specified preference weight vector (User-specified Weights). This means developers can precisely specify "I want helpfulness weight 0.7, harmlessness 0.3," and the algorithm guarantees the final model converges to this specific Pareto optimal solution, rather than just any Pareto solution.

#### **4.2.3 Experimental Verification**

Experiments on multiple model families like Qwen 3, Llama 3, and Gemma 3 show that RACO draws a better Pareto Front than traditional DPO (Direct Preference Optimization) and weighted reward models in multi-objective summarization and safety alignment tasks. This provides a solid mathematical tool for customizing model behavior (e.g., customizing high-safety models for children's education, high-helpfulness models for professional research).

---

## **5. Vertical Domain Architectural Innovation: Specialized Tools for Science and Creativity**

Besides general architecture evolution, this week also saw highly applicable results in specialized domain architectures.

### **5.1 MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training**

> **MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.02494-b31b1b.svg)](https://arxiv.org/abs/2602.02494) [**PDF Download**](../Resource/第五期/2602.02494.pdf)

**Source**: arXiv:2602.02494 [cs.LG, q-bio.NC] **Date**: Feb 2, 2026 **Authors**: Dulhan Jayalath, Neural Processing Lab

**Background & Breakthrough**: Non-invasive Brain-Computer Interfaces (MEG/EEG) have always been plagued by low signal-to-noise ratios and large individual differences. Previous models usually only analyzed a few seconds of brain signals. **MEG-XL** borrows from the Transformer-XL architecture in NLP, extending the processing window to an astonishing **2.5 minutes** (about 191k Tokens).

<div align="center">
  <img src="../Resource/第五期/MEG-XL-MEG-XL introduces long-context MEG pre-training.png" width="80%">
  <br>
  <em>Figure 9: MEG-XL Introduces Long-Context MEG Pre-training Mechanism (Source: Original Paper Figure 1)</em>
</div>


**Technical Details**: To process such long sequences, MEG-XL uses the **BioCodec** neural tokenizer, an encoder based on Residual Vector Quantization (RVQ) capable of compressing high-frequency EEG signals into discrete Tokens while retaining extremely high fidelity.

<div align="center">
  <img src="../Resource/第五期/MEG-XL-Overview of the MEG-XL pre-training framework.png" width="80%">
  <br>
  <em>Figure 10: MEG-XL Pre-training Framework Overview (Source: Original Paper Figure 2)</em>
</div>


**Significance**: Experiments prove that brain thought activity has long-range temporal dependencies. By "seeing" longer, MEG-XL learns universal neural representations across subjects. With only 1 hour of fine-tuning data from the target subject, its decoding accuracy reached levels that traditional methods require 50 hours of data to achieve. This has revolutionary significance for the clinical popularization of BCI (reducing calibration time).

### **5.2 PaperBanana: Automating Academic Illustration for AI Scientists**

> **PaperBanana: Automating Academic Illustration for AI Scientists**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.23265-b31b1b.svg)](https://arxiv.org/abs/2601.23265) [**PDF Download**](../Resource/第五期/2601.23265.pdf)

**Source**: arXiv:2601.23265 [cs.CL, cs.CV] **Date**: Jan 30, 2026 **Authors**: Dawei Zhu et al.

**Background & Breakthrough**: Scientific illustrations (like method flowcharts) require rigorous logical topological structures, which is a weakness of ordinary text-to-image models (like DALL-E 3). **PaperBanana** introduces **NAG (Native Architecture for Graphs)**.

**Technical Details**: NAG is not an external GNN, but directly modifies the internal attention mechanism of the LLM. By recalibrating Positional IDs and Attention Masks, NAG allows the LLM to understand graph node connections and hierarchical relationships "natively" like understanding natural language grammar. Combined with a specialized rendering agent, PaperBanana can automatically generate vector-level academic illustrations compliant with NeurIPS/ICLR standards.

<div align="center">
  <img src="../Resource/第五期/paperbanana-Examples of methodology diagrams and statistical plots generated by PaperBanana.png" width="80%">
  <br>
  <em>Figure 11: PaperBanana Generated Methodology Diagrams and Statistical Plots Examples (Source: Original Paper Figure 1)</em>
</div>


### **5.3 PrevizWhiz: Generative Previsualization for Film Industry**

> **PrevizWhiz: Generative Previsualization for Film Industry**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.03838-b31b1b.svg)](https://arxiv.org/abs/2602.03838) [**PDF Download**](../Resource/第五期/2602.03838.pdf)

**Source**: arXiv:2602.03838 [cs.HC, cs.AI] **Date**: Feb 4, 2026 **Authors**: Erzhen Hu et al.

**Innovation**: Targeting the "Previsualization" (Previz) stage in early film production, PrevizWhiz proposes a hybrid workflow. It allows directors to use extremely experimental, rough 3D geometry (Blocking) to define spatial relationships and camera movements, and then uses Generative Video Models to render these rough models into stylized high-fidelity video clips. This bridges the gap between traditional static storyboards and expensive 3D full-process production, representing a typical case of Generative AI landing in professional creative workflows.

<div align="center">
  <img src="../Resource/第五期/PrevizWhiz-PrevizWhiz supports an authoring workflow that transforms rough 3D scenes and video remixing into generative videos..jpg" width="80%">
  <br>
  <em>Figure 12: PrevizWhiz Creation Workflow Transforming Rough 3D Scenes to Generative Videos (Source: Original Paper Figure 1)</em>
</div>


---

## **6. Summary and Outlook: The Dawn of Systemic Intelligence**

Reviewing the frontier progress of computer science from January 29 to February 5, 2026, we see a clear shift from "Single-point Breakthrough" to "System Reconstruction."

*   **PixelGen** tells us that the authenticity of physical space is superior to the compression efficiency of latent space, heralding the arrival of the **"Post-Compression Era."**
*   **Vision-DeepResearch** and **AgentDoG** tell us that the future of agents lies in **"Depth"**—deep research capabilities and deep self-diagnostic capabilities, rather than simple breadth.
*   **RLAnything** and **RACO** tell us that the endgame of optimization lies in **"Dynamic Balance"**—whether it is the co-evolution of internal system components or the Pareto trade-off between multiple objectives.
*   **MEG-XL** and **PaperBanana** demonstrate the power of **"Domain Specialization"**—by specifically modifying infrastructure (such as long windows, graph attention), general models can explode with amazing potential in specific scientific fields.

These studies jointly depict a more refined, autonomous, and controllable AI future. For researchers, the opportunity at the moment may no longer lie in training larger models, but in re-examining those accustomed architectural assumptions and finding new levers at the level of system design.
