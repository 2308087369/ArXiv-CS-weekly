[English](./第七期_en.md) | [中文](./第七期.md)

# ArXiv Weekly - Issue 7, February 2026

> **Keywords of the Week**: Mobile-O, Edge Native Multimodal, UPipe, Headwise Chunking, EditCtrl, Local Video Editing, Nemotron-Terminal, Terminal Agent, TTT-KV, Linear Attention, ProactiveMobile, Proactive Intelligence Benchmark, Pass@k Paradox, Alignment Tax

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Edge & Proactive Intelligence: From Reactive to Proactive](#2-edge--proactive-intelligence-from-reactive-to-proactive)
  - [2.1 Mobile-O: Unified Mechanism for Edge Native Multimodal Understanding and Generation](#21-mobile-o-unified-mechanism-for-edge-native-multimodal-understanding-and-generation)
  - [2.2 ProactiveMobile: A Comprehensive Benchmark for Proactive Intelligence on Mobile Devices](#22-proactivemobile-a-comprehensive-benchmark-for-proactive-intelligence-on-mobile-devices)
- [3. Agent Evolution: Data Engineering](#3-agent-evolution-data-engineering)
  - [3.1 Nemotron-Terminal: Data Engineering for Scaling Terminal Agent Capabilities](#31-nemotron-terminal-data-engineering-for-scaling-terminal-agent-capabilities)
- [4. Infrastructure & Theory: Breaking Memory Walls & Alignment Paradoxes](#4-infrastructure--theory-breaking-memory-walls--alignment-paradoxes)
  - [4.1 Untied Ulysses (UPipe): Activation Memory Optimization for Long-Context Parallelism](#41-untied-ulysses-upipe-activation-memory-optimization-for-long-context-parallelism)
  - [4.2 TTT with KV Binding: Mathematical Reconstruction of Test-Time Training](#42-ttt-with-kv-binding-mathematical-reconstruction-of-test-time-training)
  - [4.3 Pass@k Optimization: Metric Conflict Mechanisms in RLHF and Alignment Evaluation](#43-passk-optimization-metric-conflict-mechanisms-in-rlhf-and-alignment-evaluation)
- [5. Generative Models: Fine-Grained Control & Real-Time Editing](#5-generative-models-fine-grained-control--real-time-editing)
  - [5.1 EditCtrl: Spatiotemporal Attention Disentanglement for Generative Video Editing](#51-editctrl-spatiotemporal-attention-disentanglement-for-generative-video-editing)

---

## **1. Abstract**

The current academic evolution presents three highly representative core directions. First, in the field of **edge computing and mobile intelligence**, researchers are breaking traditional cloud dependency, pushing for the real-time landing of native multimodal architectures on mobile devices through highly decoupled feature fusion mechanisms and proactive intent inference frameworks. Second, addressing the bottlenecks of **complex reasoning and massive information processing**, academia is exploring new paths like "Context Parallelism" to break through the context pollution and VRAM "Memory Wall" of large models from the organizational form of computing architecture. Finally, in **model alignment and post-training theory**, researchers are fundamentally reflecting on the underlying partial differential equation mechanisms of Test-Time Training (TTT) and multi-sample generation metrics (like Pass@k), revealing endogenous gradient conflicts between multiple traditional optimization objectives.

---

## **2. Edge & Proactive Intelligence: From Reactive to Proactive**

### **2.1 Mobile-O: Unified Mechanism for Edge Native Multimodal Understanding and Generation**

> **Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.20161-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.20161) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第七期/2602.20161.pdf) [![GitHub](https://img.shields.io/badge/GitHub-Mobile--O-181717.svg?style=flat-square&logo=github)](https://github.com/Amshaker/Mobile-O)

**Source**: arXiv:2602.20161 [cs.CV] **Date**: Feb 2026 **Authors**: MBZUAI & CMU

#### **Edge Native Multimodal Architecture**

<div align="center">
  <img src="../Resource/第七期/Mobile-O-Overview of the proposed unified multimodal post-training pipeline.png" width="80%">
  <br>
  <em>Figure 1: Mobile-O Architecture Overview: Integrates FastVLM-0.5B understanding module with SANA-600M generation module, achieving efficient feature alignment via MCP (Source: Original Paper)</em>
</div>
<div align="center">
  <img src="../Resource/第七期/Mobile-O-Qualitative comparison of text-to-image generation (left) and visual understanding (right) across unified multimodal models.png" width="80%">
  <br>
  <em>Figure 2: Mobile-O Performance Showcase: Mobile-O achieves superior results with fewer parameters in text-to-image (left) and visual understanding (right) tasks (Source: Original Paper)</em>
</div>

Deploying modern large foundation models on edge devices has long been constrained by extremely limited High Bandwidth Memory (HBM) capacity and computational power bottlenecks. Traditional solutions often employ two distinct model architectures to handle visual understanding tasks (e.g., VQA, OCR) and high-resolution image generation tasks separately, or rely heavily on cloud APIs for remote inference. This fragmented architecture not only leads to intolerable communication latency but also exposes serious data security risks in an era of increasingly strict privacy protection. Addressing this long-standing pain point, **Mobile-O** proposes a groundbreaking unified architecture designed specifically for mobile and edge deployment.

The core innovation of the Mobile-O architecture lies in its deep integration and extreme lightweight design of the Vision-Language-Diffusion paradigm. Its base understanding module, **FastVLM-0.5B**, cleverly combines an efficient FastViT visual encoder with a Qwen2-0.5B language backbone containing only 500 million parameters. On the generation side, Mobile-O abandons traditional massive diffusion models in favor of a lightweight linear Diffusion Transformer (DiT) architecture called **SANA-600M-512**, supplemented by a Variational Autoencoder (VAE) deeply optimized for 512x512 resolution.

However, the study's most disruptive contribution is its designed **Mobile Conditioning Projector (MCP)**. Traditional multimodal models often rely on dense Cross-Attention mechanisms with quadratic computational complexity when aligning visual and text features. In contrast, MCP is a tiny connection component containing only about 2.4 million parameters. It dynamically regulates feature flow distribution through innovative **Layerwise Feature Fusion** technology combined with Temperature-scaled Weights. Furthermore, MCP deeply applies Depthwise-separable 1D Convolutions and Efficient Channel Attention mechanisms, compressing the computational complexity of cross-modal alignment from quadratic to linear, thereby achieving ultra-low latency feature mapping.

#### **Experimental Verification**

In terms of experimental completeness and performance verification, the study not only conducted quantitative evaluations on standard academic benchmarks but also provided rare on-device deployment test data on real mobile hardware. Mobile-O demonstrated comprehensive capabilities surpassing its peers while maintaining an extremely limited parameter scale.

| Evaluation Metric | Mobile-O Performance | Mainstream Competitors (e.g., Show-O, JanusFlow) | Hardware Deployment & Resource Consumption |
| :---- | :---- | :---- | :---- |
| Total Parameters | 1.6B | Significantly lower than mainstream unified models | Total memory usage strictly < 2GB |
| 512x512 Image Gen | ~3s / image | Inference speed up to 11x faster than baseline | Native run on Apple iPhone |
| Visual Understanding Latency | ~0.4s / interaction | Competitors often need cloud support or have higher latency | Native run on Apple iPhone |
| GenEval Score | 74% | Leads Show-O by 5%, JanusFlow by 11% | Standard offline academic evaluation |
| Visual Understanding Avg (7 tasks) | Leads baselines | Avg accuracy leads by 15.3% and 5.1% | Standard offline academic evaluation |

Mobile-O's successful engineering practice profoundly demonstrates that the bottleneck of edge multimodal intelligence is not an insurmountable absolute parameter scale, but the alignment efficiency of cross-modal features in low-dimensional manifold spaces. Through precise architectural pruning and the application of depth-separable operators, resource-constrained edge devices are fully capable of closing the loop on complex "perception-understanding-generation" tasks independently.

---

### **2.2 ProactiveMobile: A Comprehensive Benchmark for Proactive Intelligence on Mobile Devices**

> **ProactiveMobile: A Comprehensive Benchmark for Boosting Proactive Intelligence on Mobile Devices**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.21858-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.21858) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第七期/2602.21858.pdf)

**Source**: arXiv:2602.21858 [cs.AI] **Date**: Feb 2026

#### **Proactive Intelligence & Benchmark Construction**

<div align="center">
  <img src="../Resource/第七期/ProactiveMolbile-A comparison of proactive and reactive paradigm in mobile agents.png" width="80%">
  <br>
  <em>Figure 3: ProactiveMobile Benchmark Overview: Contrasts traditional reactive paradigm with proactive paradigm, showing an evaluation system covering four heterogeneous context dimensions (Source: Original Paper)</em>
</div>

Currently, most open-source and closed-source Multimodal Large Language Models (MLLMs), despite showing amazing aptitude in visual QA, logical reasoning, and coding, still rigidly follow a "User Command -> Model Reactive Response" paradigm in their deep interaction logic with humans. When facing visual occlusion, ambiguous context, or unclear instructions, human assistants usually ask proactively to clarify intent or preemptively take action based on past experience. However, current models severely lack this forward-looking intervention capability. In the mobile device scenario, which highly emphasizes user experience, "passive waiting" has become the core obstacle restricting AI assistants from evolving into true OS-level Agents.

The **ProactiveMobile** benchmark breaks the limitation of traditional benchmarks that only evaluate static images or single text QA. The study strictly defines "proactive tasks" mathematically: models must be able to infer **Latent User Intent** across four heterogeneous context dimensions of mobile devices—Visual State, Sensor Data, Interaction Logs, and Temporal Cues. Based on this, models cannot just give text replies but must autonomously decide to accurately generate and dispatch an executable function sequence from a massive function pool containing 63 real mobile APIs.

To truly reflect the extreme complexity of the physical world and the ambiguity of intent, the research team not only constructed 3,660 instances covering 14 typical daily usage scenarios but also introduced an open evaluation system supporting multi-correct answer annotations. Notably, to ensure benchmark rigor, a team of 30 human experts conducted multiple rounds of deep auditing on all data, eliminating any flawed samples with factual fallacies, broken logic chains, or unexecutable actions.

#### **Mainstream Model Evaluation**

To measure model performance, the study adopted an advanced metric system considering temporal dynamics and action accuracy, and conducted a ruthless and real systematic "baseline" test on current top-tier mainstream large models.

| Frontline Model Array | Core Benchmark Success Rate | Deep Analysis of Proactive Intelligence Dimension |
| :---- | :---- | :---- |
| GPT-4 (SOTA Commercial Closed Source) | 7.39% | Extremely lacks proactive intent inference, performs bottom in ambiguous context, tends to conservative replies rather than API calls |
| o1 (Closed Source Strong in Reasoning) | 15.71% | Strong logical reasoning allows capturing partial intent, but still cannot fully break the frame to adapt to proactive trigger mechanisms |
| Qwen2.5-VL-7B-Instruct (Fine-tuned) | 19.15% | After high-intensity fine-tuning specifically for proactive generation tasks, broke the passive defense line, achieving best current benchmark performance |

The extremely dismal performance of current state-of-the-art commercial models on the ProactiveMobile benchmark (success rates generally below 16%) profoundly and coldly exposes a fatal systematic blind spot in the currently widely adopted RLHF training paradigm in the industry. To avoid generating harmful outputs or hallucinations, current alignment algorithms are levying an extremely high **"Safety Alignment Tax"** on models. Models are over-aligned into a "hyper-safe, absolutely passive, don't ask don't tell" rigid form, causing them to completely lose the ability to take risks and preemptive actions based on ambiguous or incomplete contexts. However, the data of Qwen2.5-VL topping the list through specific fine-tuning proves that proactivity is not an insurmountable technical chasm.

---

## **3. Agent Evolution: Data Engineering**

### **3.1 Nemotron-Terminal: Data Engineering for Scaling Terminal Agent Capabilities**

> **On Data Engineering for Scaling LLM Terminal Capabilities**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.21193-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.21193) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第七期/2602.21193.pdf) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Nemotron--Terminal-yellow.svg?style=flat-square&logo=huggingface)](https://huggingface.co/collections/nvidia/nemotron-terminal)

**Source**: arXiv:2602.21193 [cs.AI] **Date**: Feb 2026 **Authors**: NVIDIA

#### **Systematic Data Engineering**

<div align="center">
  <img src="../Resource/第七期/On Data Engineering-Overview of Terminal-Task-Gen.png" width="80%">
  <br>
  <em>Figure 4: Terminal-Task-Gen Pipeline: Includes Seed-based and Skill-based task generation, trajectory filtering, and curriculum learning mechanisms (Source: Original Paper)</em>
</div>

Large Language Models (LLMs) have shown immense potential as general Digital Agents in controlling computer terminals and automating command-line interactions. However, compared to natural language conversation and static code generation, endowing models with "Terminal Capabilities"—accurately executing multi-step OS commands and self-correcting based on system feedback—has been a major industry challenge. The core bottleneck lies in the extreme scarcity of high-quality terminal interaction data containing real environmental feedback and multi-step logical reasoning.

This research did not focus on piling up model neuronal architectures but poured all innovative resources into a scalable synthetic task generation pipeline named **Terminal-Task-Gen**. This highly engineered system aims to discard the lengthy and inefficient multi-agent coordination phase, achieving ultra-fast synthesis of massive data by simplifying environment verification logic. The core mechanism of Terminal-Task-Gen contains two driving engines: one is complex task extension based on real-world Seed-based scenarios, and the other is targeted combinatorial synthesis for specific underlying computer skills (Skill-based, e.g., complex grep regex matching, sed stream editing, network port configuration).

To cope with the inevitable "hallucination" feedback of large models when generating these long-period trajectories, the system introduced extremely strict **Trajectory Filtering** and cleaning strategies, forcibly eliminating any samples with internal logical contradictions or those causing serious kernel errors or state inconsistencies upon execution. Furthermore, in the data mixing and model feeding stage, researchers strictly implemented a **Curriculum Learning** mechanism, enabling the model to absorb the intrinsic logic of terminal instructions in a gradient from simple to deep during long-context training. On this basis, using the advanced NVIDIA NVFP4 (4-bit floating point precision) mixed training format, the research team successfully trained and open-sourced the **Nemotron-Terminal** model family and the corresponding massive synthetic dataset **Terminal-Corpus** on the Qwen3 series base.

#### **Benchmark Testing**

Through comprehensive testing on the industry-recognized most challenging Terminal-Bench 2.0 benchmark, this research proved with detailed data that excellent data engineering can fully bridge the innate gap brought by model parameter scale.

| Model & Scale | Baseline Score (Terminal-Bench 2.0) | Score After Terminal-Corpus Fine-tuning | Absolute Gain |
| :---- | :---- | :---- | :---- |
| Qwen3-8B Base | 2.5% | 13.0% | +10.5% |
| Qwen3-14B Base | 4.0% | 20.2% | +16.2% |
| Qwen3-32B Base | 3.4% | 27.4% (Matches/Surpasses huge proprietary models) | +24.0% |

The systematic success of Nemotron-Terminal sends a strong signal to the entire AI academic and engineering community: in vertical fields heavily reliant on environmental interaction like embodied intelligence and OS control, merely pursuing the Scaling Laws of model parameters is inefficient; the Scaling Laws of data quality and synthesis pipelines have risen to become the dominant factor.

---

## **4. Infrastructure & Theory: Breaking Memory Walls & Alignment Paradoxes**

### **4.1 Untied Ulysses (UPipe): Activation Memory Optimization for Long-Context Parallelism**

> **Untied Ulysses: Memory-Efficient Context Parallelism via Headwise Chunking**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.21196-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.21196) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第七期/2602.21196.pdf)

**Source**: arXiv:2602.21196 [cs.DC] **Date**: Feb 2026

#### **Headwise Chunking & Out-of-Order Scheduling**

<div align="center">
  <img src="../Resource/第七期/Untied UIysses-Illustration of (a) DeepSpeed-Ulysses and (b) UPipe designs.png" width="80%">
  <br>
  <em>Figure 5: UPipe Mechanism Diagram: Breaking the Memory Wall via Headwise Chunking and Out-of-Order Scheduling (Source: Original Paper)</em>
</div>

With the surge in multimodal input and long document processing needs, the context length modern Transformer models need to support is jumping from 100K to millions or even tens of millions. However, geometric sequence length growth hits an indestructible "Activation Memory Wall" in distributed training. Current dominant Context Parallelism technologies, like DeepSpeed Ulysses or Ring Attention, solve the compute bottleneck of single nodes but do not fundamentally reduce the absolute memory consumption of intermediate tensors inside attention layers.

**UPipe** proposes a parallel scheduling strategy named **"Headwise Chunking"**. Traditional methods tend to compute all Attention Heads simultaneously in one forward pass, causing HBM to accommodate huge intermediate activation tensors for all heads at a certain instant. UPipe breaks this norm by serializing the massive matrix multiplication of the entire attention layer into multiple discrete computation stages. In each micro-execution stage, the compute unit only processes a small subset (a Chunk) of the attention head set. Since different attention heads are mathematically orthogonal and independent in the self-attention mechanism, this chunked execution does not destroy the model's mathematical equivalence but allows the system to immediately release and reuse the VRAM space occupied by these intermediate tensors after completing a Chunk's computation.

Additionally, to be compatible with the Grouped Query Attention (GQA) architecture widely adopted by current large models, UPipe designed an extremely complex **Out-of-Order Scheduling**. Since multiple Query heads share the same set of Key/Value heads in GQA, simple sequential chunking would cause K/V tensors to be repeatedly and inefficiently globally communicated across compute nodes. UPipe's out-of-order scheduling ensures that Query heads sharing the same K/V pair are compactly combined within the same micro-compute stage, thereby minimizing the redundant overhead of All-to-All communication while fully preserving GQA's intrinsic VRAM advantage.

#### **Memory & Throughput Optimization**

Experimental results fully demonstrate the powerful efficacy of UPipe as "plug-and-play" middleware, achieving amazing memory reduction without sacrificing training throughput.

| Evaluation Metric | UPipe Performance | DeepSpeed Ulysses Performance |
| :---- | :---- | :---- |
| Attention Layer Intermediate Tensor Memory Reduction (32B Model) | Up to 87.5% peak reduction | Baseline, linear explosion with sequence length |
| Single Node (8xH100) Max Context Length | 5M Tokens (Llama3-8B) | OOM at ~3M Tokens |
| 16xH100 Cluster Max Context Length | 8M Tokens (Qwen3-32B) | Far below this capacity |

By performing deep chunking in the Attention Head dimension of the attention matrix, UPipe perfectly bypasses the physical upper limit of single GPU HBM, thoroughly shattering the long-standing "memory-speed" trade-off iron law in long-sequence model training.

---

### **4.2 TTT with KV Binding: Mathematical Reconstruction of Test-Time Training**

> **Test-Time Training with KV Binding Is Secretly Linear Attention**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.21204-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.21204) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第七期/2602.21204.pdf)

**Source**: arXiv:2602.21204 [cs.LG] **Date**: Feb 2026

#### **From Memorization Paradox to Linear Attention**

In LLM research over the past two years, Test-Time Training (TTT) and its variants with KV Binding were widely considered advanced implicit mechanisms for "Online Meta-learning" via internal loops or deep "Memorization". However, this paper launches a fundamental subversion of this mainstream perception with extremely rigorous mathematical derivation and detailed ablation experiments.

The paper's breakthrough lies in proving from the underlying perspective of algebra and calculus: a large class of seemingly extremely complex TTT architectures, even if wrapped in non-linear MLPs, complex weight normalization, adaptive learning rates, and momentum update equations, can be losslessly rewritten and reduced to a **Learned Linear Attention Operator** with enhanced representation capability in terms of absolute mathematical equivalence.

To thoroughly dismantle the traditional "memorization" hypothesis, the research team dissected and revealed the **"Memorization Paradox"** hidden in the TTT mechanism. The most shocking is "The Gradient Ascent Anomaly": researchers tried to forcibly reverse the gradient descent process used to minimize loss in the inner loop to a "gradient ascent" process maximizing loss. The result was not a model collapse, but a miraculous maintenance of performance. This phenomenon irrefutably proves: the sign reversal of internal weights is actually directly absorbed and canceled out by the newly learned Value Projection matrix outside the network. Based on these profound mathematical insights, the authors proposed an extremely pure TTT layer **Fully Parallel Formulation**.

#### **Performance Boost**

| Model Variant Ablation Trajectory | Language Modeling Perplexity (Lower is Better) | Inference Throughput Boost (Relative to Baseline) |
| :---- | :---- | :---- |
| Original Baseline TTT (w/ Momentum, MLP, etc.) | 16.43 | 1.0x (~4.30M tokens/s) |
| Variant 6 (Stripped of all redundancy, reduced to pure Linear Attention) | 16.80 (Extremely tiny decay) | **4.0x Explosion (Up to 124.6M tokens/s)** |

This research ruthlessly but elegantly pulls the TTT mechanism, wrapped in the cloak of complex optimization algorithms, back into the mathematical framework of classic linear attention mechanisms and Kernel Methods. By completely eliminating the sequence computation dependency brought by internal loop optimization, this method releases huge hardware parallelization potential, causing inference throughput to skyrocket by a full 4 times.

---

### **4.3 Pass@k Optimization: Metric Conflict Mechanisms in RLHF and Alignment Evaluation**

> **Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.21189-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.21189) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第七期/2602.21189.pdf)

**Source**: arXiv:2602.21189 [cs.LG] **Date**: Feb 2026

#### **Implicit Reweighting & Gradient Conflict**

In the post-training phase of LLMs, the Pass@k metric has become an extremely mainstream objective function in reinforcement learning optimization due to its excellent multi-sample exploration evaluation capability. However, in engineering practice, researchers repeatedly encounter a paradox: as the Pass@k metric rises, the performance of single direct response (Pass@1) suffers severe degradation.

This paper discovers from the mathematical level that Policy Gradient Update for the Pass@k objective is mathematically equivalent to an extremely unbalanced **"Implicit Reweighting"** of the probability distribution of all input Prompts in the model's internal parameter space. Under this mechanism, for those edge and obscure prompts that are extremely difficult for the model to solve (success rate close to 0), their gradient weights contributed in each backpropagation are non-linearly amplified.

Even more fatally, researchers defined a class of specific sample sets called **"Negatively Interfering Prompts"**. Due to the entanglement characteristics of the high-dimensional feature representation space inside deep neural networks, when model parameters are forcibly twisted to cater to those abnormally amplified difficult samples, the gradient direction of network weight updates undergoes severe spatial deflection, directly opposing the optimal gradient direction for optimizing high-frequency simple samples. The paper strictly proves the existence of a critical **"Phase Transition Threshold"** from the perspective of functional analysis. Once the sampling number $k$ exceeds this threshold, the synthetic gradient vector of the Pass@k objective and the Pass@1 objective will form an obtuse angle greater than 90 degrees mathematically. This means any strategy gradient update step aimed at improving Pass@k will, by mathematical necessity, lead to a dimensionality reduction strike on Pass@1 performance.

This finding issues a warning to the industry: blindly deploying Pass@k RLHF algorithms that encourage divergent exploration without scrutiny will destroy the system's reliable single-response capability on the most routine and core tasks.

---

## **5. Generative Models: Fine-Grained Control & Real-Time Editing**

### **5.1 EditCtrl: Spatiotemporal Attention Disentanglement for Generative Video Editing**

> **EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2602.15031-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.15031) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第七期/2602.15031.pdf)

**Source**: arXiv:2602.15031 [cs.CV] **Date**: Feb 2026

#### **Local & Global Decoupled Control**

<div align="center">
  <img src="../Resource/第七期/EditCtrl Video Diffusion Framework Overview.png" width="80%">
  <br>
  <em>Figure 8: EditCtrl Flowchart: Parallel collaboration of Local Context Module and Global Context Embedder (Source: Original Paper)</em>
</div>
<div align="center">
  <img src="../Resource/第七期/EditCtrl-A Real-time Generative Video Editing Pipeline.png" width="80%">
  <br>
  <em>Figure 9: EditCtrl Real-time Editing Pipeline: Supporting complex prompt-guided editing and dynamically allocating compute resources (Source: Original Paper)</em>
</div>

In the field of Generative AI, text-prompt-based video generation and editing are on the eve of an explosion. However, existing end-to-end video diffusion models often struggle when facing fine-grained semantic editing tasks. Traditional "Full-Attention" mechanisms not only bring disastrous computational latency but also cause generated content to easily destroy unedited regions in original frames.

**EditCtrl** proposes a novel computational resource dynamic allocation architecture, achieving local and global decoupling of video inpainting. Its theoretical cornerstone is the realization that in video editing tasks, high-frequency semantic transformations usually only occur in extremely small spatial localities, while low-frequency cues maintaining temporal consistency are distributed in the global background. Based on this principle, the research team decoupled the video control process into two highly specialized parallel control networks:
1.  **Local Context Module**: Activated only within user-defined mask regions, encoding foreground features at full resolution to ensure highest fidelity of texture details.
2.  **Global Context Embedder**: Downsamples the vast background video to a low-dimensional latent space, extracting macro cues like lighting, motion trajectory, and tone, injecting them into the local generation process via cross-attention modulation.

Additionally, to support seamless long video real-time generation, EditCtrl introduces a forward autoregressive content propagation mechanism, intelligently warping current frame edit results and inferring them to future frames via optical flow technology, achieving true real-time interaction.

#### **Efficiency & Quality Evaluation**

| Core Evaluation Dimension | EditCtrl Architecture Performance | Traditional Full-Attention Generation Framework Baseline |
| :---- | :---- | :---- |
| Compute Denoising Latency (Efficiency) | Up to 10x faster calculation speed relative to baseline | Extremely slow, usually requires tens of seconds to minutes for frame-by-frame rendering |
| Compute Resource Allocation Strategy | Dynamic Adaptive (Resource consumption proportional to user modification area) | Static Fixed (Resource consumption proportional to global video resolution) |
| Complex Editing Scenario Control | Perfectly supports multi-prompt decoupled editing & multi-region arbitrary masks | Easily prone to semantic confusion, target object identity loss, or background collapse |

The EditCtrl framework challenges the mindset that "high-quality video generation must rely on dense global attention" from the algorithmic bottom layer, enabling cinematic AI effects that could originally only be rendered offline on cloud servers to be realized in "What You See Is What You Get" real-time editing on local workstations.

---
