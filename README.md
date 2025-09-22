# Large Language Model Safety

This repository contains the engineering implementations behind my two research works on defending Large Language Models (LLMs) against jailbreak attacks.  
The focus is on **building practical, lightweight, and effective safety mechanisms** that can be directly integrated into real-world inference pipelines.

---

## 🚀 Projects

### 1. [Early Exit Generation Defense](https://arxiv.org/pdf/2408.11308)
![EGO.drawio.png](eeg_defender%2FEGO.drawio.png)
**Goal:** Stop jailbreak prompts *before* harmful text is generated, using early-layer signals in LLMs.

**Highlights**
- Built **prototype classifiers** over intermediate transformer embeddings to detect harmful intent in early layers.
- Designed a **plug-and-play early-exit gate** that can refuse responses in real time, without retraining or modifying the base model.
- Implemented a scalable pipeline with configurable hyper-parameters (`α`, `t`) for different models.
- Benchmarked across **3 open-source LLMs** (Vicuna, LLaMA-2, Guanaco).

**Experiments**
- Evaluated on **10 jailbreak attacks** (e.g., GCG, AutoDAN, TAP, Pair, AIM, Refusal Suppression).
- Compared against **5 baselines** (PPL, ICD, SafeDecoding, RA-LLM, etc.).
- Metrics: Attack Success Rate (ASR), Benign Answer Rate (BAR).
- Results: Achieved **~85% reduction in ASR**, significantly outperforming baselines while maintaining high BAR.

**Features**
- Efficient **classifier training over model embeddings**.
- **Inference-time optimization** for minimal latency overhead.
- **Robust evaluation harness** with multiple adversarial prompting frameworks.
- Clear **metric-driven validation** and reporting.

---

### 2. [Safety Knowledge Neurons & SafeTuning](https://arxiv.org/pdf/2509.01631)

**Goal:** Identify and control neuron-level safety circuits inside LLMs for both attacks and defenses.
![Training (3).jpg](safety_neurons%2FTraining%20%283%29.jpg)
**Highlights**
![Flow.drawio.png](safety_neurons%2FFlow.drawio.png)
- Built tools to **project MLP activations into vocabulary space**, exposing interpretable “safety neurons.”
- Implemented **causal neuron calibration**, altering less than **0.3% of parameters** to reliably flip refusal/compliance behavior.
- Developed **SafeTuning**, a fine-tuning method that selectively strengthens refusal behavior using neuron-level insights.
- Designed modular APIs to apply neuron steering in both PyTorch and Hugging Face transformer stacks.

**Experiments**
- Benchmarked **attack mode** (neuron calibration) → achieved **>97% attack success rate**, surpassing embedding-based baselines.
- Benchmarked **defense mode (SafeTuning)** → consistently reduced ASR while preserving utility across **5 datasets**.
- Conducted **sparsity studies**: proved tuning <0.1% of safety-critical neurons yields strong results without degrading general capabilities.

**Features**
- **Low-level model surgery**: fine-grained neuron identification, projection, and activation manipulation.
- **Efficient fine-tuning** targeting sparse neuron subsets.
- **Causal interpretability tooling** for debugging and visualization.
- **Robust evaluation** using keyword matching, LLM-as-a-judge, and multi-task benchmarks.

---

## 🛠️ Skills

- **LLM Safety Engineering**: Practical defenses (gates, fine-tuning, neuron steering).
- **Systems Engineering**: Designed scalable pipelines for detection and defense with minimal overhead.
- **Machine Learning Engineering**: Trained/evaluated classifiers, built adversarial test harnesses, tuned hyper-parameters.
- **Experimentation & Benchmarking**: Ran controlled evaluations across multiple models, attacks, and datasets with clear metrics.
- **Open-Source Development**: Modular, documented, reproducible code (PyTorch, Hugging Face).
- **Interdisciplinary Collaboration**: Bridging research ideas (neurons, interpretability) into deployable engineering solutions.

---

## 📊 Repository Structure
```
.  
├── eeg_defender/ # Early Exit Defense implementation  
│ ├── classifiers/ # Prototype classifier training and evaluation  
│ ├── inference/ # Early-exit generation pipeline    
│ └── benchmarks/ # Attack/defense evaluation scripts  
│  
├── safety_neurons/ # Neuron-level analysis and SafeTuning  
│ ├── neuron_finder/ # Safety neuron identification tools  
│ ├── calibration/ # Neuron editing + causal control  
│ └── safetuning/ # Targeted fine-tuning pipelines  
│
└── README.md
```



## 📎 References
```bibtex
@article{eegdefender2024,
    title={Defending against Jailbreak through Early Exit Generation of Large Language Models},
    author={Chongwen Zhao, Zhihao Dou, Kaizhu Huang},
    journal={arXiv preprint arXiv:2408.11308},
    year={2024}
}

@article{safetyneurons2025,
    title={Unraveling LLM Jailbreaks Through Safety Knowledge Neurons},
    author={Chongwen Zhao, Kaizhu Huang},
    journal={arXiv preprint arXiv:2509.01631},
    year={2025}
}
```
