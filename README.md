# Mamba-Project

README-style summary:

This project investigates the architectural shift from attention-based Transformers to Selective State Space Models (SSMs), using Mamba as the case study. Transformers face two core bottlenecks — O(n²) compute during training and linearly-growing KV-cache memory during inference — which limit their scalability on long sequences. Mamba addresses this through input-dependent (selective) state-space parameterization and a hardware-aware parallel scan, achieving linear-time computation with a constant-size hidden state.

The repo includes:

A literature review tracing the S4 → Mamba → Mamba-2 → hybrid (Jamba) lineage
A benchmarking harness comparing GPT-2 and Mamba-130M-HF on inference time, peak GPU memory, and perplexity across sequence lengths (256–2048 tokens)
Isolated core-algorithm tests (full attention vs. prefix-linear scan) to empirically validate theoretical complexity claims
Visualizations of time/memory scaling behavior for both architectures
