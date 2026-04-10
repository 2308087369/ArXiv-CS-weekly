[English](./第十三期_en.md) | [中文](./第十三期.md)

# ArXiv Weekly - 2026 April Issue 13

> Keywords this week: Knowledge Delivery, Hierarchical Parallel Agent, Aggregation Strategy Optimization, Agent Security Benchmark, Multi-Agent Reinforcement Learning, Visual Reflection Mechanism, 3D Engine Narrative, Trajectory-Aware Grading

## Table of Contents

- [1. Summary](#1-summary)
- [2. Efficiency Optimization and Zero-Bit Delivery](#2-efficiency-optimization-and-zero-bit-delivery)
  - [2.1 Knowledge Packs: KV Cache Injection and State Space Steering](#21-knowledge-packs-kv-cache-injection-and-state-space-steering)
  - [2.2 StableTTA: Mathematical Breakthrough in Aggregation Strategy Optimization](#22-stabletta-mathematical-breakthrough-in-aggregation-strategy-optimization)
- [3. Agent Systematic Architecture and Auditing](#3-agent-systematic-architecture-and-auditing)
  - [3.1 InfoSeeker: Hierarchical Parallel Agent Framework](#31-infoseeker-hierarchical-parallel-agent-framework)
  - [3.2 Claw-Eval: A New Benchmark for Trajectory Visibility Evaluation](#32-claw-eval-a-new-benchmark-for-trajectory-visibility-evaluation)
  - [3.3 AgentHazard: Security Defenses for Computer-Use Agents](#33-agenthazard-security-defenses-for-computer-use-agents)
- [4. Multimodal and Multi-Agent System Evolution](#4-multimodal-and-multi-agent-system-evolution)
  - [4.1 MARL-GPT: Cross-Domain Foundation Model for Multi-Agent Reinforcement Learning](#41-marl-gpt-cross-domain-foundation-model-for-multi-agent-reinforcement-learning)
  - [4.2 V-Reflection: Deep Correction via Visual Reflection Mechanism](#42-v-reflection-deep-correction-via-visual-reflection-mechanism)
  - [4.3 StoryBlender: 3D Engine-Driven Narrative Consistency](#43-storyblender-3d-engine-driven-narrative-consistency)
- [5. Conclusion](#5-conclusion)

---

## 1. Summary

Research in the first week of April 2026 reveals that academia is transforming models from "black-box inference engines" into "systematized agents" equipped with physical perception, hierarchical planning, and auditable execution trajectories. This issue highlights: in terms of efficiency, zero-token KV Cache injection and test-time aggregation optimization significantly alleviate computational constraints; regarding agent architecture, hierarchical topologies decouple reasoning from execution, while novel safety benchmarks emphasize the auditing of execution trajectories that are locally legal but globally harmful; in perception and evolution, the multi-agent domain is undergoing a model-scaling transition, and visual tasks achieve higher physical and narrative coherence through dynamic reflection in latent reasoning and 3D engine feedback mechanisms.

---

## 2. Efficiency Optimization and Zero-Bit Delivery

### 2.1 Knowledge Packs: KV Cache Injection and State Space Steering

> Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection

- **Motivation**: In large-scale Retrieval-Augmented Generation (RAG), a persistent bottleneck is that the forward-pass token cost of retrieved content grows linearly with the number of facts. This not only consumes massive computational power but also increases latency due to the accumulation of long historical contexts.
- **Methodology**: The study leverages the "KV-Prefix Equivalence" principle in the Transformer architecture. By performing offline forward passes on massive facts, it extracts and serializes their KV caches into "Knowledge Packs." During inference, by directly injecting pre-computed states into the inference engine, the query sequence interacts directly with the facts' hidden states. It also reveals the second-order effect of "Value-Space Steering," changing generation styles via vector arithmetic on cache Values.
- **Results**: On Qwen3-8B and Llama-3.1-8B, the retrieval latency for 5,000 facts dropped from 50-200ms to ~6ms, achieving zero token cost for single-step retrieval and saving 95.3% of tokens in 5-step cumulative retrieval, while maintaining byte-level consistency with standard concatenation inference.
- **Insights**: Without modifying model parameters, this approach establishes a dual-channel mode of "knowledge delivery + behavior steering," providing a novel pathway for real-time online alignment and extremely low-cost fact retrieval.

### 2.2 StableTTA: Mathematical Breakthrough in Aggregation Strategy Optimization

> StableTTA: Training-Free Test-Time Adaptation that Improves Model Accuracy on ImageNet1K to 96%

- **Motivation**: In computer vision, improving performance on ImageNet-1K usually relies on adding data or scaling up parameters. In Test-Time Adaptation (TTA), different aggregation strategies (like hard voting, soft voting, and logit averaging) often produce conflicting and mutually exclusive predictions when handling sparse logit spaces.
- **Methodology**: The study systematically identifies and proves that aggregation conflicts arise from the nonlinear and non-bijective nature of the Softmax and indicator functions. It proposes a training-free hybrid method that introduces stability constraints during image preprocessing and enforces strategy alignment during logit post-processing, eliminating instability.
- **Results**: Without any additional training data, it boosted 33 common models to over 95% Top-1 accuracy. For instance, MobileNetV3 surpassed a standard-sized Vision Transformer (ViT) in accuracy, using less than 5% of the parameters while saving 97% in VRAM and 89.1% in GFLOPs.
- **Insights**: Physical architectural stacking is not the only panacea for performance. Mathematical optimization at the model inference logic level (like resolving aggregation conflicts) holds immense potential, paving the way for deploying SOTA systems on low-power mobile devices.

---

## 3. Agent Systematic Architecture and Auditing

### 3.1 InfoSeeker: Hierarchical Parallel Agent Framework

> InfoSeeker: A Scalable Hierarchical Parallel Agent for Large-Scale Information Seeking
> [![arXiv](https://img.shields.io/badge/arXiv-2604.02971-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2604.02971) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十三期/InfoSeeker.pdf)

- **Motivation**: Traditional ReAct-based agents often fail when facing large-scale tasks that require aggregating dozens of webpages, due to context window saturation, broken logic chains, and high latency from sequential execution.
- **Methodology**: Adopting the principle of "near-decomposability," it builds a three-tier topology to decouple reasoning depth from execution width: 1) Strategic Host (maintains state summaries and plans, avoiding raw data overflow); 2) Domain-Specific Managers (break down instructions into fine-grained tasks); 3) Parallel Workers (execute tools concurrently based on the Model Context Protocol).
- **Results**: It overpowered existing closed-source systems on WideSearch and BrowseComp-zh benchmarks. The success rate improved relatively by 66.7% over baselines, with a 3-5x speedup. Its context isolation mechanism reduced the cascading error rate by 40%, greatly enhancing robustness against network anomalies like webpage hangs.
- **Insights**: Completing large complex tasks no longer relies solely on expanding the context length of monolithic models; a distributed agent architecture based on the MapReduce pattern can more effectively improve system execution throughput and fault tolerance.

<div align="center">
  <img src="../Resource/第十三期/InfoSeeker_Figure1.png" width="80%">
  <br>
  <em>Figure: The hierarchical parallel agent architecture of InfoSeeker.</em>
</div>

### 3.2 Claw-Eval: A New Benchmark for Trajectory Visibility Evaluation

> Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents
> [![arXiv](https://img.shields.io/badge/arXiv-2604.06132-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2604.06132) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十三期/Claw-Eval.pdf)

- **Motivation**: Existing agent evaluations rely too heavily on being "results-oriented," leading to evaluation blind spots that ignore potential violations like inefficient retries and illegal access during execution, creating a false sense of prosperity.
- **Methodology**: It introduces the concept of "trajectory-aware grading," collecting evidence through three independent channels: Execution Traces, Audit Logs, and Environment Snapshots. Covering 300 tasks and 2,159 fine-grained grading criteria, it systematically breaks down the evaluation of completion, safety, and robustness.
- **Results**: Evaluating 14 frontier models revealed that traditional black-box evaluations (checking only the final output) missed 44% of safety violations and 13% of robustness failures. Claw-Eval achieved a 100% detection rate, effectively puncturing the illusion of superficial task success rates.
- **Insights**: As agents enter industrial-level deployment, "procedural justice" and intermediate state evaluations will become standard. Models must not only complete tasks successfully but also ensure their execution trajectories are trustworthy, safe, and stable.

<div align="center">
  <img src="../Resource/第十三期/Claw-Eval_Figure1.png" width="80%">
  <br>
  <em>Figure: The Claw-Eval trajectory-aware grading system.</em>
</div>

### 3.3 AgentHazard: Security Defenses for Computer-Use Agents

> AgentHazard: A Benchmark for Evaluating Harmful Behavior in Computer-Use Agents


- **Motivation**: As agents gain capabilities for persistent operations and environment management, security risks have shifted from outputting "harmful speech" to "dynamic execution." Complex execution trajectories mean that combinations of seemingly legal single-step operations could lead to severe security vulnerabilities.
- **Methodology**: It proposes an attack design philosophy of "locally legal, globally harmful," constructing a dynamic risk benchmark with 2,653 instances across 10 risk categories and 10 attack strategies. It focuses on testing trajectory-dependent vulnerabilities like persistent malware establishment and sensitive information exfiltration.
- **Results**: Testing mainstream model frameworks like Qwen3, Kimi, and GLM revealed that Claude Code (GLM-4.6) had an Attack Success Rate (ASR) of 82.90%. The ASR difference for the same model under different frameworks exceeded 16 percentage points, exposing the failure of current alignment techniques against complex, long-horizon operations.
- **Insights**: The design of System Prompts, routing logic, and permission boundaries plays a decisive role in overall security. Agent defense must upgrade from simple content interception to fine-grained causal auditing of underlying system calls.

---

## 4. Multimodal and Multi-Agent System Evolution

### 4.1 MARL-GPT: Cross-Domain Foundation Model for Multi-Agent Reinforcement Learning

> MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning
> [![arXiv](https://img.shields.io/badge/arXiv-2604.05943-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2604.05943) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十三期/MARL-GPT.pdf)

- **Motivation**: Traditional Multi-Agent Reinforcement Learning (MARL) is highly specialized in domains like StarCraft and fails to achieve cross-task transferability.
- **Methodology**: Abandoning customized architectures, it adopts a large-scale offline imitation learning approach based on GPT. It uniformly maps observations, actions, and value functions into token streams. It integrates over 1.5 billion trajectory data from SMACv2, Google Research Football, etc., and designs an observation encoder supporting variable agent numbers and permutation invariance.
- **Results**: MARL-GPT matched or surpassed SOTA models across three different simulators. For example, it achieved a 54% win rate in SMACv2 (baseline 14-30%), approached the expert score ceiling in the football environment, and demonstrated the potential for zero-shot transfer via contextual fine-tuning on maps of unseen scales.
- **Insights**: Reinforcement learning is undergoing an "NLP-like scaling transformation to unified Transformers." The introduction of general sequence modeling capabilities makes it possible to build foundational decision models capable of executing complex collaborative strategies across diverse environments.

<div align="center">
  <img src="../Resource/第十三期/MARL-GPT_Figure1.png" width="80%">
  <br>
  <em>Figure: MARL-GPT utilizing a unified architecture for cross-domain offline imitation learning.</em>
</div>

### 4.2 V-Reflection: Deep Correction via Visual Reflection Mechanism

> V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators


- **Motivation**: Multimodal Large Language Models (MLLMs) often hallucinate when executing fine-grained perception tasks because they cannot re-examine visual inputs based on their reasoning state.
- **Methodology**: It proposes a "think before you look" dynamic reflection mechanism, transforming model latent states into Dynamic Probes. Through a two-stage distillation strategy (explicit spatial anchoring BCM and visual latent space distillation DAC), spatial parsing capabilities are internalized, enabling the model to autonomously locate evidence in the global visual graph based on its thought trajectory.
- **Results**: It significantly improved precision across six perception benchmarks, especially when handling 4K images and subtle distinctions. V-Reflection achieved 81.2% accuracy on HRBench-4K (a 6.7% increase over the baseline) with zero additional architectural overhead or token latency during inference.
- **Insights**: Transforming visual information from a "static background" into a "dynamic participant" in the reasoning process proves that end-to-end latent reasoning flows are more suitable for high-density spatial perception tasks than explicit external tool calls.

### 4.3 StoryBlender: 3D Engine-Driven Narrative Consistency

> StoryBlender: Inter-Shot Consistent and Editable 3D Storyboard with Spatial-temporal Dynamics


- **Motivation**: In visual video or image synthesis, cross-shot character identity drift and geometric consistency have always been technical bottlenecks.
- **Methodology**: It reframes storyboard generation as a hierarchical multi-agent planning process within a 3D engine. It constructs a four-tier "Continuity Memory Graph" (including storyboard outlines and asset tables) to decouple global assets (identity) from local variables (action, lighting), and provides a feedback loop for self-correction via engine physical verification.
- **Results**: Compared to traditional diffusion models, the system not only completely eliminated cross-shot identity drift but also allowed creators to perform spatial-level re-editing and modifications directly in the generated 3D scenes, demonstrating native 3D control and physical accuracy far surpassing existing methods.
- **Insights**: Integrating an external deterministic computation engine (3D simulator) as a feedback loop for generative AI is a crucial leap, pushing generative models from random artistic tools toward rigorous industrial pre-visualization pipelines.

---

## 5. Conclusion

Summarizing this issue's literature, systematized evolution under computational resource constraints has become the main theme. On the one hand, "zero-bit" and "training-free" mathematical logic optimizations show that mining the geometric efficiency of existing parameter manifolds yields higher returns than blindly scaling up models. On the other hand, whether in hierarchically decoupled large parallel agent frameworks, dynamically reflecting visual probing systems, or narrative generation integrated with 3D engines, researchers are building "governance structures" that transcend single-network forward propagation. Future competition will no longer be confined to the brain's computational capacity but will shift toward a comprehensive battle over the stability of perception-execution loops, the interpretability of process trajectories, and architectural fault tolerance.
