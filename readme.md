# ✨𝗠𝗢𝗠✨: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models 

### 🔎Overview

**Memory-efficient Offloaded Mini-sequence Inference (𝗠𝗢𝗠)  partitions critical layers into smaller "mini-sequences" and integrates seamlessly with KV cache offloading.**

<img src=".\doc\images\MSIAch.png" width="240"/>   <img src=".\doc\images\memory_speed_tradeoff.png" width="480"/>





### ✅ Features

- 🦙 **Supports LLaMA 3, Qwen2.5, and Mistral-Nemo**
- 💾 **Reduces peak memory usage by over 50% on average**
- 📈 **Extends maximum context length from 155k to 455k tokens on a single A100 80GB GPU**
- 🎯 **Preserves output accuracy — generates identical results**
- ⚡ **Minimal computational overhead, with accelerated last-layer processing**

<p align="center">
<img src=".\doc\images\max_context_extended.png" width="400"/>
</p>

<!-- <br/><br/> -->


**The method drastically reduces prefill memory consumption, eliminating it as the longstanding dominant memory bottleneck during inference. This breakthrough fundamentally changes research priorities, redirecting future efforts from prefill-stage optimizations to improving decode-stage residual KV cache efficiency.**

<p align="center">
<img src=".\doc\images\fullCompare.png" width="600"/>
</p>

### 🧪Testing

```bash
pip install -r requirements.txt 
```

**[Checkout the notebook for 𝗠𝗢𝗠]()**

Or run the python scripts in this directory for comparison between different configuration, and visualize the results


