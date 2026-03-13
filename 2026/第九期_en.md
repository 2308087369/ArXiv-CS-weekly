[English](./第九期_en.md) | [中文](./第九期.md)

# ArXiv Weekly - Issue 9, March 2026

> **Keywords**: InternVL-U, Unified Multimodal, MM-Zero, Zero-Data Self-Evolution, Unsupervised RLVR, Model Collapse, Spatial-TTT, Streaming Spatial Intelligence, Thinking to Recall, Computational Buffer, CoT-Control, Inference Controllability, OpenClaw-RL, Next-State Signals, EVATok, Adaptive Tokenization

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Lightweight Democratization of Unified Multimodal Models: Technical Path and Semantic Alignment of InternVL-U](#2-lightweight-democratization-of-unified-multimodal-models-technical-path-and-semantic-alignment-of-internvl-u)
- [3. Breakthrough in Zero-Data Self-Evolution: MM-Zero and Three-Role Collaborative Logic](#3-breakthrough-in-zero-data-self-evolution-mm-zero-and-three-role-collaborative-logic)
- [4. Expanding Boundaries of Unsupervised RL: "Model Collapse Step" in RLVR](#4-expanding-boundaries-of-unsupervised-rl-model-collapse-step-in-rlvr)
- [5. The Rise of Streaming Spatial Intelligence: Spatial-TTT and Test-Time Training Memory](#5-the-rise-of-streaming-spatial-intelligence-spatial-ttt-and-test-time-training-memory)
- [6. Physical Mechanisms Behind Chain-of-Thought: Computational Buffer Effect and Factual Priming](#6-physical-mechanisms-behind-chain-of-thought-computational-buffer-effect-and-factual-priming)
- [7. The Controllability Dilemma of Inference: Safety and Concealment in CoT Monitoring](#7-the-controllability-dilemma-of-inference-safety-and-concealment-in-cot-monitoring)
- [8. Real-Time Closed Loop of Agent Learning: OpenClaw-RL and Next-State Signals](#8-real-time-closed-loop-of-agent-learning-openclaw-rl-and-next-state-signals)
- [9. Efficiency Revolution in Video Generation: EVATok and Adaptive Token Allocation](#9-efficiency-revolution-in-video-generation-evatok-and-adaptive-token-allocation)
- [10. Conclusion and Future Trends](#10-conclusion-and-future-trends)

---

## **1. Abstract**

Between March 6 and March 13, 2026, the arXiv Computer Science (CS) section witnessed a profound transformation regarding the underlying architecture and evolutionary logic of Large Language Models (LLMs). The literature released this week not only made significant progress in the unification of multimodal understanding and generation but also demonstrated unprecedented depth in reinforcement learning verification, self-evolutionary frameworks, and the deconstruction of inference mechanisms. Current research trends are shifting from purely pursuing the "Scaling Laws" of parameter size to a refined exploration of computational efficiency, logical controllability, and unsupervised self-improvement capabilities. By screening hundreds of newly published papers during this period, this report highlights eight landmark studies that collectively map out a critical juncture where artificial intelligence is crossing from "imitating human data" to "autonomous logical discovery."

---

## **2. Lightweight Democratization of Unified Multimodal Models: Technical Path and Semantic Alignment of InternVL-U**

> **InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2603.09877-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.09877) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第九期/InternVL-U.pdf) [![GitHub](https://img.shields.io/badge/GitHub-InternVL-181717.svg?style=flat-square&logo=github)](https://github.com/OpenGVLab/InternVL)

**Source**: arXiv:2603.09877 [cs.CV] **Date**: March 2026 **Authors**: Shanghai AI Laboratory

#### **Core Innovation Mechanisms and Underlying Principles**

<div align="center">
  <img src="../Resource/第九期/InternVL-U_Figure1.png" width="80%">
  <br>
  <em>Figure 1: InternVL-U Unified Architecture Overview: Integrating an MLLM backbone for understanding with a specialized MMDiT head for generation, enabled by the UniReason data synthesis pipeline.</em>
</div>

In the field of vision-language research this week, the InternVL-U report released jointly by Shanghai AI Laboratory and multiple institutions has attracted widespread attention. The core goal of this study is to solve the inherent trade-off problem in Unified Multimodal Models (UMMs) when integrating understanding, reasoning, generation, and editing functions. Traditional models often have to sacrifice the depth of semantic understanding when pursuing strong image generation capabilities, or vice versa. InternVL-U, through a lightweight framework with only 4 billion parameters, challenges previous large unified models with hundreds of billions of parameters, proving the huge potential of modular design and decoupled visual representation in improving efficiency.

InternVL-U's innovation is first reflected in its heterogeneous modular architecture. The research team did not adopt a single, undifferentiated multimodal Transformer, but insisted on modality-specific encoding scaffolds. The model initialization is based on the InternVL 3.5 backbone network and integrates a dedicated generation head based on Multimodal Diffusion Transformer (MMDiT). The basic rationale behind this design is driven by "semantic density" differences: text is highly compressed and semantically dense, while raw visual patches are sparse and redundant. By leaving semantic reasoning tasks in a unified backbone space and assigning pixel-level synthesis tasks to a specialized generation head, InternVL-U achieves optimized configuration of computational resources, avoiding the backbone network being burdened by low-level perceptual pixel reconstruction tasks.

#### **Experimental Verification and Efficiency Advantages**

In terms of experimental design, InternVL-U demonstrates extremely high completeness. Researchers built a comprehensive data synthesis pipeline called UniReason 1.0, specifically targeting high semantic density tasks such as precise text rendering, scientific geometric reasoning, and complex humor understanding. To ensure the quality of synthetic data, the pipeline introduces the Chain of Thought (CoT) paradigm, transforming vague editing instructions into executable steps containing logical constraints, thereby achieving deep intent alignment.

| Evaluation Dimension | InternVL-U (4B) | BAGEL (14B) | Improvement/Comparison |
| :---- | :---- | :---- | :---- |
| **Generation Task Accuracy** | Leading | Basic | > 30% |
| **Editing Task Fidelity** | Excellent | Medium | Significant Gain |
| **Reasoning Benchmark (MMMU)** | 53.4% (Qwen3 Base) | ~50.2% | +3.2% |
| **Model Parameters** | 4B | 14B | Efficiency Gain >3x |

The emergence of InternVL-U heralds the "democratization" of unified multimodal models, meaning that smaller research teams or terminal devices can also run intelligent systems with full capabilities. This path through three-stage progressive training—from basic understanding inheritance to multimodal generation acquisition, and then to complex editing fine-tuning—provides an efficient reference template for moving towards Artificial General Intelligence (AGI) in the future.

---

## **3. Breakthrough in Zero-Data Self-Evolution: MM-Zero and Three-Role Collaborative Logic**

> **MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2603.09206-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.09206) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第九期/MM-Zero.pdf) [![GitHub](https://img.shields.io/badge/GitHub-MM--Zero-181717.svg?style=flat-square&logo=github)](https://github.com/NVIDIA/MM-Zero)

**Source**: arXiv:2603.09206 [cs.CV] **Date**: March 2026 **Authors**: NVIDIA

#### **Core Innovation Mechanisms and Underlying Principles**

<div align="center">
  <img src="../Resource/第九期/MM-Zero_Figure1.png" width="80%">
  <br>
  <em>Figure 2: MM-Zero Zero-Data Self-Evolution Framework: Illustrating the collaborative closed-loop of Questioner, Coder, and Solver roles, achieving cold start via procedural generation.</em>
</div>

On March 10, 2026, the MM-Zero framework proposed by NVIDIA and collaborators completely broke the dependence of Vision Language Models (VLMs) on external training data. Although previous studies have proven that Large Language Models can improve reasoning capabilities through Self-play, due to the introduction of the visual modality, VLMs usually require at least some initial seed images to start the evolutionary process. MM-Zero achieves true "Zero-Data Self-Evolution" by introducing a reinforcement learning mechanism based on procedural generation.

The major innovation of MM-Zero in principle is extending the traditional "Questioner-Answerer" two-role architecture to a "Questioner-Coder-Solver" three-role system. In this closed-loop system, all roles are initialized by the same base VLM. The Questioner is responsible for curating abstract visual concepts and designing challenging multimodal questions; the Coder translates these concepts into executable code (such as SVG or Python graphics scripts), thereby rendering precise visual scenes without an external image library; finally, the Solver performs multimodal reasoning and gives answers by analyzing these synthetic images.

This process utilizes the "Reinforcement Learning with Verifiable Rewards (RLVR)" paradigm. The Coder's performance is evaluated by whether the code runs successfully and whether the generated image matches the semantic description; the Solver is tested on both easy and hard levels of questions, where "easy questions" serve as a baseline for semantic correctness, while "hard questions" are used to generate pseudo-labels to drive policy updates through a majority voting mechanism.

#### **Experimental Verification and Performance Gains**

Experimental results show that through three rounds of iteration, MM-Zero increased the Solver's average accuracy on mainstream multimodal benchmarks such as MMMU and ChartQA by up to 5.1%.

| Iteration Stage | Visual Concept Complexity | Solver Accuracy (Qwen3-VL-8B) | Core Gain Area |
| :---- | :---- | :---- | :---- |
| **Initial State** | Simple Geometry/Basic OCR | 50.7% | Basic Understanding |
| **Round 1** | Compositional Graphics/Logical Relations | 52.1% | Spatial Reasoning |
| **Round 3** | Scientific Charts/Complex Math | 54.1% | Visual Math Reasoning |

MM-Zero's experiments proved the importance of "Difficulty-Balanced Reward" through ablation studies. Without this dynamic adjustment based on the Goldilocks principle (i.e., tasks should be neither too hard nor too easy), the Questioner tends to produce extremely simple tasks leading to performance stagnation, or produce hallucinated tasks that cannot be rendered by code, leading to training failure.

---

## **4. Expanding Boundaries of Unsupervised RL: "Model Collapse Step" in RLVR**

> **How Far Can Unsupervised RLVR Scale LLM Training?**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2603.08660-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.08660) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第九期/Unsupervised-RLVR.pdf)

**Source**: arXiv:2603.08660 [cs.LG] **Date**: March 2026

#### **Core Innovation Mechanisms and Underlying Principles**

<div align="center">
  <img src="../Resource/第九期/Unsupervised-RLVR_Figure1.png" width="80%">
  <br>
  <em>Figure 3: Model Collapse in Unsupervised RLVR: Showing the "inverted U-shape" curve of performance rise and fall under intrinsic rewards, contrasted with extrinsic rewards.</em>
</div>

As reinforcement learning takes center stage in model post-training, a core question arises: How far can unsupervised reinforcement learning push model performance? "How Far Can Unsupervised RLVR Scale LLM Training?" released this week gives a profound qualitative and quantitative answer to this question through a rigorous theoretical framework and large-scale experiments.

Researchers categorize existing unsupervised RLVR methods into "Intrinsic Rewards" and "Extrinsic Rewards". Intrinsic rewards rely on the model's own signals (such as majority vote consistency or output entropy), while extrinsic rewards utilize external verification mechanisms (such as code compilers or self-verification asymmetry). The study reveals that all methods based on intrinsic rewards are essentially performing a "Distribution Sharpening" operation, i.e., constantly reinforcing high-probability responses in the model's initial distribution. Although this can bring obvious performance gains in the early stages of training, due to the lack of ground truth guidance, the model will inevitably move towards collapse, manifested as increased output repetitiveness and loss of diversity.

One of the most valuable insights proposed in this paper is the "Model Collapse Step" metric. Experiments found that the robustness of different model families in reinforcement learning is highly correlated with the strength of their initial priors.

| Model Family | Collapse Step (Steps) | Performance Gain (GT Gain) | Conclusion |
| :---- | :---- | :---- | :---- |
| **Llama-3.1-8B** | 40 | +1.01% | Weak Prior, Prone to Collapse |
| **Qwen2.5-7B** | 160-245 | 4-7% | Medium Stability |
| **Qwen3-8B-Base** | 383 | +17.08% | Strong Prior, Huge RL Potential |

The study further analyzed the impact of engineering parameters on collapse, reaching a counter-intuitive conclusion: increasing the sampling number (N) per iteration actually accelerates model collapse. This is because more samples reinforce the confidence of majority voting, making the model lock onto potentially incorrect preferred answers faster. In contrast, extrinsic methods utilizing "Self-Verification Asymmetry" (i.e., verifying an answer is often easier than generating it) show potential to surpass this collapse ceiling.

---

## **5. The Rise of Streaming Spatial Intelligence: Spatial-TTT and Test-Time Training Memory**

> **Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training**
>
> [![Project](https://img.shields.io/badge/Project-Website-blue.svg?style=flat-square)](https://liuff19.github.io/Spatial-TTT/)

**Source**: arXiv/Project [cs.CV] **Date**: March 2026 **Authors**: Tencent Hunyuan & Tsinghua

#### **Core Innovation Mechanisms and Underlying Principles**

<div align="center">
  <img src="../Resource/第九期/Spatial-TTT_Figure1.png" width="80%">
  <br>
  <em>Figure 4: Spatial-TTT Hybrid Streaming Architecture: Combining Sliding Window Attention (SWA) with Test-Time Training (TTT) layers, utilizing dynamic Fast Weights for long-term spatial memory.</em>
</div>

In the field of video understanding, maintaining continuous cognition of 3D space in large-scale, long-duration streaming data is a long-standing challenge. Spatial-TTT, launched this week by the Tencent Hunyuan team and Tsinghua University, proposes a highly forward-looking solution: using Test-Time Training (TTT) to transform model parameters into dynamic spatial memory.

The principle of Spatial-TTT lies in breaking the linear overhead of traditional Transformers for long context dependencies. It designs a hybrid architecture that alternates TTT layers with traditional Sliding Window Attention (SWA) layers in a 3:1 ratio. Inside the TTT layer, the model does not directly store historical tokens, but learns and organizes spatial evidence from the continuously incoming video stream online through a small set of trainable parameters called "Fast Weights". Whenever a new frame enters, the model performs a tiny weight update. This update not only captures the current position of the target but also learns the object's motion trajectory and geometric correspondence through 3D spatiotemporal convolution layers.

#### **Experimental Verification and Physical Significance**

Experimental results strongly support the superiority of this method. In the VSI-Bench test for egocentric videos, Spatial-TTT demonstrated strong stability in object recall and spatial counting tasks, and its memory growth is sub-linear.

| Video Length (Frames) | Spatial-TTT Memory Usage | Qwen3-VL-2B Memory Usage | Performance Retention |
| :---- | :---- | :---- | :---- |
| **256** | ~4.2 GB | ~5.5 GB | 100% |
| **1024** | ~6.8 GB | > 11.0 GB | 98.5% |
| **2048** | ~8.1 GB | OOM | 96.2% |

The physical significance of this architecture is that it endows the model with a decoupling similar to human "transient memory" and "long-term spatial cognition". By constantly optimizing fast weights in thousands of frames of boundless streaming data, Spatial-TTT can build a panoramic view of complex indoor scenes, which is of immeasurable value for embodied intelligence (Robotics) and real-time augmented reality applications.

---

## **6. Physical Mechanisms Behind Chain-of-Thought: Computational Buffer Effect and Factual Priming**

> **Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2603.09906-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.09906) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第九期/Thinking-to-Recall.pdf)

**Source**: arXiv:2603.09906 [cs.CL] **Date**: March 2026 **Authors**: Google Research

#### **Core Innovation Mechanisms and Underlying Principles**

<div align="center">
  <img src="../Resource/第九期/Thinking-to-Recall_Figure1.png" width="80%">
  <br>
  <em>Figure 5: Mechanism of CoT Unlocking Parametric Knowledge: Comparing direct answering, random text buffering, and logical reasoning in activating model latent knowledge.</em>
</div>

The study "Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs" published by Google Research this week presents challenging insights into the basic scientific question of why Chain-of-Thought (CoT) improves model accuracy. It is generally believed that CoT works by decomposing complex problems into simple steps, but when dealing with "single-hop" factual questions that do not require decomposition, CoT can still significantly improve the model's retrieval success rate, which is intuitively difficult to explain.

Through a series of controlled experiments, the study identified two key driving mechanisms. The first is the "Computational Buffer Effect". Researchers found that even if the model is required to generate random text or "Dummy tokens" completely unrelated to semantics, the model's performance in subsequently answering factual questions also improves. This shows that the model uses these extra reasoning tokens as a "computational scratchpad space", performing some parallel computation independent of specific word meanings in the latent space, thereby better aligning its internal parameter weights.

The second mechanism is "Factual Priming". During the reasoning process, the model spontaneously builds a "semantic bridge" to the final answer by generating relevant background facts. This Generative Self-retrieval can activate dormant knowledge points within the model. However, this mechanism is a double-edged sword: research shows that if hallucinations (i.e., false facts) appear in the reasoning path, the probability of the final answer being wrong increases significantly.

| Experimental Condition | pass@1 Accuracy (SimpleQA) | pass@100 Potential | Mechanism Explanation |
| :---- | :---- | :---- | :---- |
| **Direct Answer** | 20.6% | 40.6% | Zero Reasoning, Direct Retrieval |
| **Random Text Reasoning** | 24.1% | 48.2% | Using Only Computational Buffer Effect |
| **Logical Reasoning (CoT)** | 30.2% | 63.8% | Buffer Effect + Factual Priming |

Through audits of 13 frontier models, the study found that reasoning indeed significantly expands the boundaries of the model's parametric knowledge, making certain answers that were originally "unreachable" possible.

---

## **7. The Controllability Dilemma of Inference: Safety and Concealment in CoT Monitoring**

> **Reasoning Models Struggle to Control their Chains of Thought**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2603.05706-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.05706) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第九期/CoT-Control.pdf)

**Source**: arXiv:2603.05706 [cs.AI] **Date**: March 2026 **Authors**: OpenAI & NYU

#### **Core Innovation Mechanisms and Underlying Principles**

<div align="center">
  <img src="../Resource/第九期/CoT-Control_Figure1.png" width="80%">
  <br>
  <em>Figure 6: CoT-Control Evaluation Results: Highlighting the significant capability gap between controlling final outputs versus controlling the Chain-of-Thought (CoT) in mainstream reasoning models.</em>
</div>

As reasoning models become popular in critical social decisions, the safety community has begun to worry whether models possess the ability of "strategic concealment". "Reasoning Models Struggle to Control their Chains of Thought", jointly released by OpenAI, NYU, and other institutions, conducted in-depth testing on this and introduced an evaluation benchmark named CoT-Control.

This study attempts to answer a fundamental safety question: If our regulatory means rely on observing the model's CoT to discover its malicious intent, can the model circumvent monitoring by manipulating the CoT? Experiments designed various constraints, such as requiring the model not to mention specific mathematical terms when answering complex math problems, or not to use certain core logical connectives when explaining logic. The results were surprising: even the most powerful reasoning models currently available (such as Claude 4.5 Sonnet or DeepSeek R1) have far lower ability to control reasoning paths than their ability to control final outputs.

| Control Object | Claude 4.5 Control Success Rate | DeepSeek R1 Control Success Rate | Industry Implication |
| :---- | :---- | :---- | :---- |
| **Final Output** | 61.9% | High | Easy to Follow External Instructions |
| **Reasoning Path (CoT)** | 2.7% | 0.1% | Reasoning Path Has Extreme Rigidity |
| **CoT with Monitoring Alert** | ~6.7% | Slightly Improved | Awareness of Monitoring Slightly Enhances Control |

This "Lack of Controllability" phenomenon can be attributed in principle to the deep coupling between reasoning and task solving. For large models, CoT is not just text displayed to humans; it is the computational process itself. Requiring a model to solve a problem without using specific core concepts is equivalent to requiring an engineer to design a bridge without using the concept of "force". Furthermore, the study found an interesting "Reasoning Effort Paradox": increasing the model's reasoning duration or thinking depth actually reduces its ability to control CoT.

---

## **8. Real-Time Closed Loop of Agent Learning: OpenClaw-RL and Next-State Signals**

> **OpenClaw-RL: Train Any Agent Simply by Talking**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2603.10165-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.10165) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第九期/OpenClaw-RL.pdf) [![GitHub](https://img.shields.io/badge/GitHub-OpenClaw--RL-181717.svg?style=flat-square&logo=github)](https://github.com/Princeton-SysML/OpenClaw)

**Source**: arXiv:2603.10165 [cs.LG] **Date**: March 2026 **Authors**: Princeton AI Lab

#### **Core Innovation Mechanisms and Underlying Principles**

<div align="center">
  <img src="../Resource/第九期/OpenClaw-RL_Figure1.png" width="80%">
  <br>
  <em>Figure 7: OpenClaw-RL Asynchronous Closed-Loop Infrastructure: Featuring four fully decoupled parallel loops for Environment Hosting, PRM Judging, Policy Training, and Inference Serving.</em>
</div>

On March 9, 2026, the OpenClaw-RL framework released by Princeton AI Lab provided a new paradigm for continuous learning of Agents. The core observation of this study is: every moment an agent interacts with the environment, whether it is a user reply, a terminal error, or a GUI state change, a kind of "Next-state signal" is generated. However, existing reinforcement learning systems often discard these valuable real-time feedbacks.

OpenClaw-RL establishes an asynchronous, decoupled industrial-grade architecture, containing four major loops: environment hosting, PRM judgment, policy training, and SGLang inference. It extracts two core types of information from next-state signals. First are Evaluative Signals, converted into numerical rewards through a Process Reward Model (PRM); second are Directive Signals, realized through "Hindsight-Guided On-Policy Distillation (OPD)".

In experiments, OpenClaw-RL demonstrated extremely strong versatility, capable of learning simultaneously in four types of tasks: personal assistant dialogue, terminal operation, software engineering (SWE), and tool calling. For personal assistants, the model can learn user personalized preferences (such as a friendlier conversational tone) in just 16 interaction turns; for general agents, it showed robustness surpassing Outcome-only reward dependence when handling long-range tasks like SWE-Bench.

---

## **9. Efficiency Revolution in Video Generation: EVATok and Adaptive Token Allocation**

> **EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2603.12267-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.12267) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第九期/EVATok.pdf)

**Source**: arXiv:2603.12267 [cs.CV] **Date**: March 2026

#### **Core Innovation Mechanisms and Underlying Principles**

<div align="center">
  <img src="../Resource/第九期/EVATok_Figure1.png" width="80%">
  <br>
  <em>Figure 8: EVATok Adaptive Video Tokenization Process: Demonstrating how a lightweight router dynamically allocates token density based on video segment complexity for efficient compression.</em>
</div>

In the field of generative video models, the EVATok framework accepted by CVPR 2026 this week sets a new efficiency benchmark for autoregressive video generation. Due to the extremely high spatiotemporal redundancy of video data, traditional fixed-length Tokenization often wastes a lot of computing resources in simple scenes (such as fixed backgrounds), while leading to blurry image quality due to insufficient tokens in complex dynamic scenes.

EVATok introduces an "Efficient Video Adaptive Tokenizer" framework. Its core principle is to introduce a lightweight Router to predict the optimal token allocation ratio for each video segment during the encoding stage. For segments containing intense action or rich texture, the router assigns dense tokens; for segments with static backgrounds, token allocation is automatically diluted. This method essentially introduces the idea of classic compression algorithms—variable bit rate coding—into the deep learning tokenization process.

Experimental results show that EVATok achieved significant performance improvements in UCF-101 and more complex video reconstruction tasks. Compared to the current strongest baseline model LARP, EVATok saves an average of 24.4% token usage while maintaining equal or even higher reconstruction quality.

| Model Scheme | Token Saving Rate | Reconstruction FVD (Lower is Better) | Deployment Advantage |
| :---- | :---- | :---- | :---- |
| **Fixed Length Baseline** | 0% | Higher | Simple Architecture, but Inefficient |
| **LARP (SOTA)** | ~10% | Excellent | High Computational Cost |
| **EVATok** | 24.4% | Current Best | Significantly Reduces Inference VRAM |

---

## **10. Conclusion and Future Trends**

In the week of early March 2026, CS research on arXiv showed a high degree of synergy, reflected in the collective questioning of "interpretability, evolvability, and efficiency of intelligence". Through InternVL-U and EVATok, we saw a refined revolution at the architectural level, achieving full modal capabilities without relying on ultra-large scale parameters through decoupling and adaptive technologies; MM-Zero and OpenClaw-RL showed us how agents can break away from dependence on static datasets and turn to an autonomous evolutionary paradigm based on execution feedback and three-role gaming.

However, deeper scientific exploration is unfolding in the fields of RLVR and CoT mechanisms. Researchers are beginning to realize that the improvement of model performance comes not only from the instillation of data but also from the effective activation of the model's internal "latent computational space". At the same time, the discovery of "Model Collapse Step" and "Lack of CoT Controllability" not only sets theoretical boundaries for understanding the limits of AI but also provides empirical evidence for safety regulation. Future technological competition will no longer be just a competition of GPU numbers, but a competition of who can more effectively use verification asymmetry, who can more precisely control reasoning-induced facts, and who can more stably allow the system to achieve closed-loop improvement in an unsupervised environment. The literature of this week is the key footnote in the prologue of this grand transformation.

---
