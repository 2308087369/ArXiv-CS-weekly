[English](./第二期_en.md) | [中文](./第二期.md)

# ArXiv Weekly - January 2026 Week 2

> **Keywords of the Week**: Continual Reconstruction, Formal Constraints, Multimodal Noise Resistance, Industrial Robustness, Edge Intelligence

## Table of Contents

- [1. Differentiation of AI Research Paradigms and Deep Water Exploration](#1-differentiation-of-ai-research-paradigms-and-deep-water-exploration)
- [2. Theoretical Breakthroughs in Language Modeling: Breaking Discrete Shackles](#2-theoretical-breakthroughs-in-language-modeling-breaking-discrete-shackles)
  - [2.1 Token Maturation: Continuous Dynamics of Autoregressive Language Models](#21-token-maturation-continuous-dynamics-of-autoregressive-language-models)
  - [2.2 SemPA: Sentence Vector Optimization Based on Semantic Preference Alignment](#22-sempa-sentence-vector-optimization-based-on-semantic-preference-alignment)
- [3. Safety Revolution in Agent Engineering: Hoare Contracts and Formal Verification](#3-safety-revolution-in-agent-engineering-hoare-contracts-and-formal-verification)
  - [3.1 ToolGate: Contract-Grounded Tool Execution Verification](#31-toolgate-contract-grounded-tool-execution-verification)
  - [3.2 BackdoorAgent: Unified Backdoor Attack Framework for LLM Agents](#32-backdooragent-unified-backdoor-attack-framework-for-llm-agents)
- [4. Unification and Noise Resistance of Multimodal Systems: Towards Industrial Robustness](#4-unification-and-noise-resistance-of-multimodal-systems-towards-industrial-robustness)
  - [4.1 Qwen3-VL: Unified Multimodal Retrieval and Ranking Framework](#41-qwen3-vl-unified-multimodal-retrieval-and-ranking-framework)
  - [4.2 Lost in the Noise: Failure of Reasoning Models Under Noise Interference](#42-lost-in-the-noise-failure-of-reasoning-models-under-noise-interference)
  - [4.3 Universal Multi-Image Editing and Unified Multimodal Models](#43-universal-multi-image-editing-and-unified-multimodal-models)
- [5. Scientific Computing and Edge Intelligence: Efficiency and Physical Constraints](#5-scientific-computing-and-edge-intelligence-efficiency-and-physical-constraints)
  - [5.1 StablePDENet: Enhancing Stability of Operator Learning](#51-stablepdenet-enhancing-stability-of-operator-learning)
  - [5.2 Re-examining Training Scale: Token Count, Energy Consumption, and Parameter Efficiency](#52-re-examining-training-scale-token-count-energy-consumption-and-parameter-efficiency)
  - [5.3 FlyPose: Robust Human Pose Estimation from High-Altitude View](#53-flypose-robust-human-pose-estimation-from-high-altitude-view)
- [6. Summary and Outlook](#6-summary-and-outlook)

---

## **1. Differentiation of AI Research Paradigms and Deep Water Exploration**

Through sorting and screening hundreds of papers published on arXiv this week, we found that the focus of research is shifting from pure parameter stacking to correcting the inherent defects of models. This correction is reflected in paradigm shifts in three core dimensions:

1. **Continuous Reconstruction of Generation Mechanisms**: The "Discrete Token" assumption that has long dominated the NLP field is being fundamentally challenged. Research represented by "Token Maturation" begins to explore "semantic development" in continuous vector space, attempting to solve the uncertainty collapse problem caused by discrete sampling. This is not just a fine-tuning of algorithms, but a redefinition of the cornerstone theory of Autoregressive Generation.
2. **Formal Constraints on Agent Interaction**: As Agents move from chat boxes to Tool Use and the physical world, traditional control flows based on Natural Language Reasoning (ReAct) appear too fragile. Papers such as "ToolGate" this week introduce Hoare Logic and Design by Contract from software engineering, marking the strong return of "Neuro-Symbolic" in engineering practice.
3. **Noise Resistance and Unification of Multimodal Systems**: In the multimodal field, the focus has shifted from "generating cool images" to "high-precision retrieval" and "complex semantic consistency". The release of the Qwen3-VL series and research on NoisyBench show that the vulnerability of current systems to contextual interference far exceeds expectations, and the industrial demand for robustness has overwhelmed the single pursuit of generation capabilities.

**2. Theoretical Breakthroughs in Language Modeling: Breaking Discrete Shackles**

The core of Natural Language Processing (NLP) has long been dominated by discrete symbol systems. However, this week's research shows that to break through the bottleneck of consistency between reasoning and generation, academia is attempting to introduce continuous dynamic systems.

### **2.1 Token Maturation: Autoregressive Language Generation via Continuous Token Dynamics**

> **Token Maturation: Autoregressive Language Generation via Continuous Token Dynamics**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.04854-b31b1b.svg)](https://arxiv.org/abs/2601.04854) [**PDF Download**](../Resource/第二期/2601.04854v1.pdf)

#### **2.1.1 Innovation and Core Overview**

This paper (arXiv:2601.04854) proposes the most theoretically ambitious architecture of the week—**Token Maturation**. Author Oshri Naparstek challenges the basic operating paradigm of Autoregressive Language Models (AR-LLM): that every generation step must immediately "collapse" into a discrete Token.

In traditional Transformers, the model outputs a probability distribution $P(x\_t|x\_{\<t})$ at time $t$, and determines $x\_t$ through Sampling or Greedy Search (Argmax). Once determined, it cannot be changed. This "Early Discretization" forces the model to make hard commitments when semantics are not yet fully formed, leading to the famous "Garden Path Effect," where once the model chooses the wrong path, subsequent generation can only patch up the error, eventually leading to hallucinations or logical collapse.

<div align="center">
  <img src="../Resource/第二期/tokenf1.png" width="90%" />
</div>
<center>Figure 1: Comparison of Immediate Commitment vs. Token Maturation. (A) Standard autoregressive decoding commits a discrete Token at every step, making early decisions irreversible. (B) Token Maturation maintains a continuous "liquid tail" token representation that evolves over time, delaying discretization until the final commitment step.</center>

The core innovation of **Token Maturation** is the introduction of a **Continuous Maturation Mechanism**. The model no longer directly outputs discrete Tokens but maintains a sequence of continuous "Embryonic Vectors". These vectors evolve and correct over multiple time steps until they "mature" and converge in vector space, before being projected back to the discrete vocabulary. This gives the model the ability to "draft mentally" like humans—pre-rehearsing and adjusting multiple future words in continuous space before speaking the next word.

#### **2.1.2 Key Principles and Mathematical Formulation**

The model models the generation process as a deterministic dynamical system.

1. Continuous Representation:
   Let the vocabulary embedding matrix be $E \in \mathbb{R}^{|V| \times d}$. The model predicts a continuous vector $z_t \in \mathbb{R}^d$ at each position $t$, rather than a categorical distribution.
2. Liquid Tail and Dynamic Update:
   The model maintains a "liquid tail" sequence $\tilde{\mathbf{z}}_{n+1:n+K}$ of length $K$, representing undetermined future Tokens. In each update step, the model calculates a new prediction vector $\hat{z}$ based on the currently committed context and the liquid tail. The update follows the following dynamic equation:

   $$
   \tilde{z}_i \leftarrow \tilde{z}_i + \alpha_i (\hat{z}_i - \tilde{z}_i)
   $$

   Where:
   * $\tilde{z}_i$ is the transient vector at position $i$ at the current moment.
   * $\hat{z}_i$ is the latest prediction for position $i$ based on the current full context.
   * $\alpha_i$ is the **Maturation Rate** or momentum coefficient, usually designed as a time-decaying function (Alpha Profile). For Tokens close to the "commitment point", $\alpha$ is close to 1, indicating significant correction by the current prediction; for distant Tokens, $\alpha$ is small, allowing them to remain ambiguous.
3. Deterministic Decoding (Geometric Argmax):
   Unlike random sampling in traditional AR models, Token Maturation only requires deterministic projection to obtain diverse and coherent text after evolution in continuous space stabilizes:

   $$
   x_{n+1} = \arg\max_{j} \langle \tilde{z}_{n+1}, e_j \rangle
   $$

   Experiments prove that this deterministic path walking in geometric space avoids the repetitive loop problems common in greedy decoding of discrete space.

#### **2.1.3 Experimental Evaluation and Summary of Pros and Cons**

* Experimental Completeness: ★★★☆☆
  The author compared it on standard language modeling benchmarks against baselines including standard Transformer AR models and Diffusion-based Language Models (Diffusion LM). Results show that without using any random sampling (Temperature=0), text generated by Token Maturation surpasses baselines in Diversity and Coherence.
* **Pros**:
  * **Eliminates Sampling Uncertainty**: Shifts the source of generation randomness from "hard sampling" to "continuous semantic drift," theoretically more consistent with human thinking processes.
  * **Computational Efficiency**: Compared to diffusion models requiring hundreds of iteration steps, this method only adds a small amount of vector calculation during the autoregressive process, with extremely low increase in inference latency.
  * **Stronger Error Correction Capability**: Since Tokens undergo multiple iterations (Maturation) before formal commitment, the model effectively uses longer-term right-side context (Future Context) to correct current left-side predictions.
* **Cons**:
  * **Memory Overhead**: Requires maintaining liquid tail vectors of length $K$, slightly higher VRAM usage than standard KV Cache mechanism.
  * **Quantization Loss**: Although vector evolution is continuous, the final step still requires projection back to the discrete vocabulary. The quantization error in this step may still cause semantic mutations in extreme cases, which is not sufficiently discussed in the paper.

#

### **2.2 SemPA: Improving Sentence Embeddings of LLMs through Semantic Preference Alignment**

> **SemPA: Improving Sentence Embeddings of LLMs through Semantic Preference Alignment**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.05075-b31b1b.svg)](https://arxiv.org/abs/2601.05075) [**PDF Download**](../Resource/第二期/2601.05075v1.pdf)

#### **2.2.1 Innovation**

Traditional Sentence Embedding models (such as BERT, SimCSE) are usually separated from generative LLMs (such as GPT-4, Llama). Directly using the last layer representation of LLMs as sentence vectors suffers from severe Anisotropy, leading to poor semantic similarity calculation results.
SemPA (arXiv:2601.05075) proposes a method to fine-tune LLMs using **Semantic Preference Alignment**. Its core innovation lies in transferring the preference optimization idea from RLHF (Reinforcement Learning from Human Feedback) to the embedding space.

<div align="center">
  <img src="../Resource/第二期/Semf1.png" width="90%" />
</div>
<center>Figure 2: Comparison of sentence embedding methods. Top: Contrastive learning for encoder-only models. Bottom: LLM sentence vector optimization based on semantic preference alignment.</center>

#### **2.2.2 Key Principles**

The method constructs "Preference Pairs" for semantic similarity.

* Given a query sentence $q$, and two candidate sentences $p^+$ (semantically more similar) and $p^-$ (semantically less similar).
* The optimization goal is to maximize the similarity between the embedding vector $v_q$ generated by the LLM and $v_{p^+}$, while minimizing the similarity with $v_{p^-}$, and introducing a constraint term to prevent the model from forgetting its generation capabilities:

  $$
  \mathcal{L} = -\log \sigma (\beta (\text{sim}(v_q, v_{p^+}) - \text{sim}(v_q, v_{p^-}))) + \lambda \mathcal{L}_{\text{gen}}
  $$

  This is actually a variant of DPO (Direct Preference Optimization) in the embedding space.

<div align="center">
  <img src="../Resource/第二期/Semf2.png" width="90%" />
</div>
<center>Figure 3: SemPA method overall flow. (a) Construct paraphrase generation preference pairs using NLI datasets. (b) Perform semantic DPO training on LLM. (c) Use PromptEOL template to obtain final sentence vectors.</center>

#### **2.2.3 Summary of Pros and Cons**

* **Pros**: Achieves "One Model for Two Uses", meaning the same LLM can perform high-quality text generation and directly output SOTA-level sentence vectors, significantly reducing deployment costs.
* **Cons**: Highly sensitive to the quality of Negative Pairs mining. If preference data contains noise, it can lead to embedding space collapse.

#

**3. Safety Revolution in Agent Engineering: Hoare Contracts and Formal Verification**

As Agents are granted permissions to execute code and call APIs, their safety issues have escalated from "outputting harmful content" to "executing harmful operations." The traditional ReAct mode (Reasoning + Acting) lacks rigid constraints. This week's paper "ToolGate" provides an industrial-grade solution for this.

### **3.1 "ToolGate: Contract-Grounded and Verified Tool Execution for LLMs"**

**Paper Index**: arXiv:2601.04688v1 [cs.AI] 07 Jan 2026
**Original PDF**: [2601.04688v1.pdf](../Resource/第二期/2601.04688v1.pdf)

#### **3.1.1 Innovation and Core Overview**

**ToolGate** (arXiv:2601.04688) is a pioneering work combining Formal Methods and Large Language Models. Existing Agent frameworks (such as AutoGPT, LangChain) treat tool calling as a black box, completely relying on LLM's probabilistic output to decide when to call and what parameters to pass. This mechanism is logically **Unverifiable**, easily leading to hallucinated calls or parameter errors.

ToolGate introduces classic **Hoare Logic** from software engineering, encapsulating each tool call as a **Contract**. By explicitly maintaining a **Symbolic State Space**, the system can verify Preconditions before tool execution and Postconditions after execution, thereby building a "Forward Execution" safety fence.

<div align="center">
  <img src="../Resource/第二期/toolgatef1.jpg" width="90%" />
</div>
<center>Figure 4: ToolGate framework overview. The framework is built on Hoare logic, formalizing the tool calling process as a series of constrained logical reasoning steps, and continuously maintaining a trusted state to verify tool calling conditions.</center>

#### **3.1.2 Key Principles: Contracts and State Evolution**

1. Symbolic World State $S$:
   Unlike LLM's fuzzy KV Cache memory, ToolGate maintains an explicit, typed key-value map $S$. For example: { "user\_authenticated": True, "file\_path": "/tmp/data.csv" }. Only verified information can enter $S$.
2. Hoare-style Contract:
   For any tool $T$, define contract $\\{P\\} T \\{Q\\}$:
   * **Precondition $P(S)$**: A logical predicate. Only when $P(S)$ is true, the LLM is allowed to call tool $T$. For example, before calling readFile, it must satisfy exists(path) && hasPermission(user, path).
   * **Postcondition $Q(S, R)$**: After the tool execution returns result $R$, it must satisfy $Q$ to update the result to state $S$. This prevents the tool from returning incorrect data or hallucinated data contaminating the Agent's cognition.
3. Interception and Feedback Mechanism:
   When the LLM attempts to initiate a call, ToolGate intercepts the request and runs the validator on the symbolic state $S$. If verification fails (e.g., precondition not met), the system directly returns structured error information to the LLM, forcing the LLM to Re-plan instead of blindly retrying.

#### **3.1.3 Experimental Data and Comparison**

* Experimental Completeness: ★★★★★
  The author conducted large-scale evaluations on the authoritative ToolBench benchmark and compared mainstream frameworks like ReAct and Chameleon.
* **Key Results**:
  * **Reliability Improvement**: In complex multi-step reasoning tasks, ToolGate's success rate is significantly higher than ReAct. Although specific values are not fully detailed in the abstract, qualitative descriptions indicate an overwhelming advantage in preventing "invalid operation sequences."
  * **State Purity**: Compared to ReAct, which often mixes outputs from incorrect tools into the context causing subsequent reasoning collapse, ToolGate ensures logical consistency of the state space.

#### **3.1.4 Summary of Pros and Cons**

* **Pros**:
  * **Verifiability**: This is the most scarce attribute in current Agent systems. It makes Agent behavior auditable and debuggable.
  * **Safety**: Physically/logically isolates LLM hallucinations from the actual execution environment, preventing destructive operations.
* **Cons**:
  * **High Engineering Cost**: Each tool requires manual or semi-automatic writing of Hoare contracts (pre/post conditions). For systems with thousands of APIs, this constitutes a huge cold start cost.
  * **Limited Flexibility**: Overly rigid contracts may limit LLM's creative solutions (Emergent Capabilities) in unforeseen scenarios.

#

**3.2 "BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents"**

**Paper Index**: arXiv:2601.04566v2 [cs.AI] 07 Jan 2026
**Original PDF**: [2601.04566v2.pdf](../Resource/第二期/2601.04566v2.pdf)

#### **3.2.1 Innovation**

While ToolGate builds defenses, BackdoorAgent (arXiv:2601.04566) reveals the vulnerability of Agents. This paper systematically divides the attack surface of Agents into three stages for the first time: **Planning**, **Memory**, and **Tool-use**.

#### **3.2.2 Key Findings**

* **Cross-Stage Triggering**: Experiments show that attackers only need to implant a backdoor trigger (Trigger) in the **Planning Stage**, and this trigger will persist and affect the final result in 43.58% of cases.
* **Memory Poisoning**: If toxin is implanted in the **Memory Stage** (such as RAG's retrieval library), its propagation success rate is as high as 77.97%.
* **Significance**: This proves that current Agent workflows lack an internal "immune system." The state verification mechanism proposed by ToolGate is exactly the effective means to defend against such Memory Attacks. The two papers constitute a complete puzzle of offense and defense research this week.

<div align="center">
  <img src="../Resource/第二期/backf1.png" width="90%" />
</div>
<center>Figure 5: BackdoorAgent Framework. BackdoorAgent exposes explicit interfaces in the planning, memory, and tool stages of the agent workflow, and provides a runtime environment supporting configurable execution, attack injection, and trajectory recording.</center>

**4. Unification and Noise Resistance of Multimodal Systems: Towards Industrial Robustness**

Multimodal Large Models (MLLM) are evolving from "image captioning" to "precise retrieval" and "anti-interference reasoning." The release of Qwen3-VL and the study of NoisyBench jointly point to a trend: **Precision and Robustness are the lifelines of the next generation of MLLMs**.

### **4.1 "Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework"**

**Paper Index**: arXiv:2601.04720v1 [cs.CL] 07 Jan 2026
**Original PDF**: [2601.04720v1.pdf](../Resource/第二期/2601.04720v1.pdf)

#### **4.1.1 Innovation and System Architecture**

The **Qwen3-VL** retrieval suite (arXiv:2601.04720) released by Alibaba Tongyi Lab represents the highest level of current Multimodal Retrieval. This work no longer fights alone but launches a complete industrial-grade pipeline of **Embedding (Recall)** + **Reranker (Re-ranking)**.

<div align="center">
  <img src="../Resource/第二期/qwenf1.png" width="90%" />
</div>
<center>Figure 6: Schematic of Unified Multimodal Representation Space. Qwen3-VL-Embedding series models map multi-source data (text, images, visual documents, and videos) into a common manifold.</center>

1. **Unified Representation Space**: The Qwen3-VL-Embedding model can map heterogeneous modalities such as text, images, document screenshots, and videos to the same high-dimensional vector space. This breaks the limitation that text retrieval and image retrieval required different models in the past.
2. **Matryoshka Representation Learning (MRL)**: This is a highly practical innovation of the model. The model is trained to contain most semantic information in the first $k$ dimensions (e.g., first 256 dimensions). This means developers can arbitrarily truncate vector dimensions (from 4096 dimensions to 64 dimensions) according to storage and bandwidth budgets during deployment, with minimal accuracy loss.
3. **Two-Stage Architecture**:
   * **Embedding Model (Dual Tower)**: Used for fast recall of massive data, supporting input up to 32k Tokens (can process long videos or long documents).
   * **Reranker Model (Cross-Encoder)**: Used for fine-grained scoring of recall results. It deeply fuses features of Query and Document through full-layer Cross-Attention, outputting precise relevance scores.

<div align="center">
  <img src="../Resource/第二期/qwenf2.png" width="90%" />
</div>
<center>Figure 7: Multi-stage training pipeline of Qwen3-VL-Embedding and Qwen3-VL-Reranker.</center>

#### **4.1.2 Experimental Data: Refreshing SOTA**

* **MMEB-V2 Leaderboard**: The Qwen3-VL-Embedding-8B model achieved a comprehensive score of **77.8** (World No. 1 as of January 8, 2026), a significant improvement over previous open-source SOTAs (such as Seed-1.6).
* **Multilingual Capability**: Supports over 30 languages, solving the pain point of weak Chinese capabilities in models like CLIP.

#### **4.1.3 Summary of Pros and Cons**

* **Pros**:
  * **Engineering Friendly**: MRL and Quantization support make this model extremely easy to deploy on resource-constrained edge devices.
  * **Comprehensiveness**: Solves both "Fast Search" (Embedding) and "Accurate Search" (Reranker) problems simultaneously.
* **Cons**:
  * **Training Details Not Fully Disclosed**: Although model weights were released, the technical report is vague about specific loss function configurations and negative sample mining strategies for Contrastive Learning.

#

### **4.2 Lost in the Noise: Failure of Reasoning Models Under Noise Interference**

> **Lost in the Noise: How Reasoning Models Fail with Contextual Distractors**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2601.07226-b31b1b.svg)](https://arxiv.org/abs/2601.07226) [**PDF Download**](../Resource/第二期/2601.07226v1.pdf)

#### **4.2.1 Innovation: NoisyBench and Inverse Scaling Phenomenon**

This paper (arXiv:2601.07226) throws a bucket of cold water on the currently hot RAG and long-context reasoning. The authors constructed **NoisyBench**, specifically to test model performance in the face of "Distractors".

<div align="center">
  <img src="../Resource/第二期/noisef1.png" width="90%" />
</div>
