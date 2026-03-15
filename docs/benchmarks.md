# Performance Benchmarks

All benchmarks are collected using `llm-server.sh bench`.

## Quantization Comparison (MiniMax M2.5)

Two-node distributed inference over Thunderbolt.

| Quant | Size | Gen (tok/s) | Prompt (tok/s) | Quality |
|-------|------|-------------|----------------|---------|
| UD-Q6_K_XL | ~181 GB | ~13 | ~195 | Highest |
| Q6_K | ~175 GB | ~13 | ~195 | Very High |
| Q5_K_M | ~155 GB | ~14 | ~210 | High |
| Q4_K_XL | ~130 GB | ~16 | ~230 | Good |

Lower quantization trades some quality for faster speed and smaller memory footprint. Q4_K_XL is the sweet spot if the model needs to fit alongside a large KV cache.

## Backend Comparison

Same model and quantization, comparing Vulkan (RADV) vs ROCm/HIP backends.

| Backend | Gen (tok/s) | Prompt (tok/s) | Notes |
|---------|-------------|----------------|-------|
| Vulkan (RADV) | ~13 | ~195 | Better generation speed |
| ROCm/HIP | ~11 | ~225 | Better prompt processing |

**Vulkan is the default** because generation speed (tok/s during output) matters more for interactive use. Prompt processing is a one-time cost per request, while generation speed determines the perceived responsiveness.

## Context Size vs Memory

| Context Size | KV Cache Quant | Approx Memory per Node | Fits in 192 GB (96 GB x2)? |
|--------------|----------------|-------------------------|-----------------------------|
| 131K | q8_0 | ~90 GB model + KV cache | Yes |
| 262K | q8_0 | Exceeds available | No |
| 262K | q4_0 | May fit with smaller model quant | Possibly |

At 131K context with q8_0 KV cache, each node uses approximately 90 GB (model shard + KV cache), fitting well within the 96 GB VRAM allocation.

## Interconnect Bandwidth

| Configuration | Bandwidth | Notes |
|---------------|-----------|-------|
| Single USB4 v2 cable (rear) | ~38 Gbps | Standard setup |
| Dual cable | Higher aggregate | Second cable on 10.0.1.0/24 subnet |
| Single USB4 v1 cable (front) | ~9.3 Gbps | Not recommended |

Single rear cable at ~38 Gbps is sufficient for the current generation speeds. The interconnect is not the bottleneck for models at ~13 tok/s generation.
