 [English](./第十一期_en.md) | [中文](./第十一期.md)
 
 # ArXiv Weekly - Issue 11 (Mar 2026)
 
 > Keywords: ultra-long context, sparse attention, document-wise RoPE, latent path injection, institutional alignment, pre-action authorization, online robot rewards, streaming multi-shot video, log semantic compression, pricing reversal
 
 ## Contents
 
 - [1. Overview](#1-overview)
 - [2. Long-Context Memory: MSA](#2-long-context-memory-msa)
 - [3. Structured Retrieval & Latent Injection: S-Path-RAG](#3-structured-retrieval--latent-injection-s-path-rag)
 - [4. Agent Safety & Institutional Alignment](#4-agent-safety--institutional-alignment)
   - [4.1 Agentic AI perspective](#41-agentic-ai-perspective)
   - [4.2 Pre-action authorization (OAP)](#42-pre-action-authorization-oap)
 - [5. Embodied Intelligence: Large Reward Models](#5-embodied-intelligence-large-reward-models)
 - [6. Streaming Video Generation](#6-streaming-video-generation)
 - [7. Systems & AI Economics](#7-systems--ai-economics)
   - [7.1 LogFold: structured log compression](#71-logfold-structured-log-compression)
   - [7.2 Pricing reversal in reasoning models](#72-pricing-reversal-in-reasoning-models)
 - [8. Closing](#8-closing)
 
 ---
 
 ## 1. Overview
 
 Three themes: scalable lifetime memory with end-to-end sparse attention; topology-aware graph retrieval with latent path injection; and deployable safety/economics with deterministic pre-action authorization, log semantic compression, and cost-transparent model selection. For embodiment, VLM-driven online rewards cut reward-engineering effort; multi-shot video moves toward interactive streaming.
 
 ---
 
 ## 2. Long-Context Memory: MSA
 
 > MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens  
 > [![arXiv](https://img.shields.io/badge/arXiv-2603.23516-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.23516) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十一期/MSA.pdf)
 
 - Design: end-to-end sparse attention with document-wise parallel RoPE + global RoPE offset; < 9% degradation when extrapolating to 100M tokens.  
 - Systems: KV cache compression and Memory Parallel for 2×A800; Memory Interleave for multi-round cross-document reasoning.  
 - Takeaway: decouple memory capacity from reasoning with compressible, schedulable latent states.
 
<div align="center">
  <img src="../Resource/第十一期/MSA_Figure1.png" width="80%">
  <br>
  <em>Figure: MSA overview and scalability (16K→100M with < 9% degradation).</em>
  <br><br>
  <img src="../Resource/第十一期/MSA_Figure2.png" width="80%">
  <br>
  <em>Figure: Memory Sparse Attention layer and document-wise RoPE.</em>
</div>

 ---
 
 ## 3. Structured Retrieval & Latent Injection: S-Path-RAG
 
 > S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop KGQA  
 > [![arXiv](https://img.shields.io/badge/arXiv-2603.23512-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.23512) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十一期/S-Path-RAG.pdf)
 
 - Retrieval: hybrid weighted k-shortest/beam/constrained random walk plus contrastive path encoder and lightweight verifier.  
 - Injection: compact soft mixture of path latents injected via cross-attention into a frozen LLM; diagnostic signals guide iterative graph edits.  
 - Result: higher answer coverage and efficiency vs. text concatenation; token-efficient and topology-aware.
 
<div align="center">
  <img src="../Resource/第十一期/S-Path-RAG_Figure1.png" width="80%">
  <br>
  <em>Figure: S-Path-RAG pipeline—path enumeration → latent injection → iterative dialogue.</em>
</div>

 ---
 
 ## 4. Agent Safety & Institutional Alignment
 
 ### 4.1 Agentic AI perspective
 
 > Agentic AI and the next intelligence explosion  
 > [![arXiv](https://img.shields.io/badge/arXiv-2603.20639-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.20639) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十一期/Agentic-AI.pdf)
 
 - View: frontier reasoning models exhibit “societies of thought”; shift from dyadic alignment to institutional alignment with role protocols and checks & balances.
 
 ### 4.2 Pre-action authorization (OAP)
 
 > Before the Tool Call: Deterministic Pre-Action Authorization for Autonomous AI Agents  
 > [![arXiv](https://img.shields.io/badge/arXiv-2603.20953-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.20953) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十一期/Before-the-Tool-Call.pdf)
 
 - Core: synchronous interception before tool execution; declarative policies; Ed25519-signed immutable audit logs.  
 - Outcome: attack success cut from 74.6% to 0% with ~53 ms median latency; complements sandboxing and model-based screening.
 
<div align="center">
  <img src="../Resource/第十一期/Before-the-Tool-Call_Figure1.png" width="80%">
  <br>
  <em>Figure: OAP authorization flow—deterministic policy evaluation before tool execution.</em>
</div>

 ---
 
 ## 5. Embodied Intelligence: Large Reward Models
 
 > Large Reward Models: Generalizable Online Robot Reward Generation with VLMs  
 > [![arXiv](https://img.shields.io/badge/arXiv-2603.16065-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.16065) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十一期/Large-Reward-Models.pdf)
 
 - Approach: LoRA-specialized Qwen3-VL-8B produces frame-level temporal contrastive, absolute progress, and completion rewards; no manual shaping.  
 - Evidence: zero-shot generalization; significant gains within 30 online RL iterations for long-horizon manipulation.
 
<div align="center">
  <img src="../Resource/第十一期/Large-Reward-Models_Fig1.png" width="80%">
  <br>
  <em>Figure: LRM overview—VLM-derived frame-level rewards guiding online RL.</em>
</div>

 ---
 
 ## 6. Streaming Video Generation
 
 > ShotStream: Streaming Multi-Shot Video Generation for Interactive Storytelling  
 > No public arXiv found yet (related lines: memory-conditioned multi-shot, streaming diffusion scheduling).  
 - Idea: next-shot autoregression with dual caches for intra-shot continuity and cross-shot consistency; supports interactive control during inference.  
 - Goal: near-real-time FPS and latency control on consumer GPUs for XR, games, and storytelling.
 
 ---
 
 ## 7. Systems & AI Economics
 
 ### 7.1 LogFold: structured log compression
 
 > LogFold: Compressing Logs with Structured Tokens and Hybrid Encoding  
 > Venue: ICSE 2026 (no public arXiv).  
 - Method: delimiter-skeleton–aware token analysis; type-specific encodings (e.g., delta/elastic for numeric).  
 - Outcome: +11.11% average compression ratio over SOTA on 16 datasets with 9.842 MB/s throughput.
 
 ### 7.2 Pricing reversal in reasoning models
 
 > The Price Reversal Phenomenon: When Cheaper Reasoning Models End Up Costing More  
 > [![arXiv](https://img.shields.io/badge/arXiv-2603.23971-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.23971) [![PDF](https://img.shields.io/badge/PDF-Download-red.svg?style=flat-square)](../Resource/第十一期/Price-Reversal.pdf)
 
 - Finding: across 8 RLMs × 9 tasks, 21.8% model pairs show rank reversals; root cause is heterogeneous, high-variance thinking tokens.  
 - Guidance: prioritize per-request costs over listed prices; enable auditable cost monitoring and dynamic routing.
 
<div align="center">
  <img src="../Resource/第十一期/Price-Reversal_Figure1.png" width="80%">
  <br>
  <em>Figure: Pricing reversal—“cheaper” list price can cost more in total.</em>
</div>

 ---
 
 ## 8. Closing
 
 MSA and S-Path-RAG advance scalable memory and structured retrieval; OAP and pricing reversal ground deployment in safety and economics. LRM and streaming video shorten paths from cognition to action and from description to rendering. Building controllable, auditable, and affordable AI systems moves us toward institutionalized, deployable general intelligence.
 
