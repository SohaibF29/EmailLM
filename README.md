# 📧 Email Writing Assistant — Llama 3.1 8B Fine-Tuned on Enron Corpus

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Spaces-fbbf24?style=for-the-badge)
![Llama 3.1](https://img.shields.io/badge/Llama_3.1-8B_Instruct-6366f1?style=for-the-badge)

> **A multi-task email generation system** that composes professional emails from subjects and generates contextual replies — powered by a QLoRA fine-tuned Llama 3.1 8B Instruct model trained on the Enron corporate email corpus.

---

## 🎯 Overview

This project fine-tunes Meta's **Llama 3.1 8B Instruct** on the [Enron Email Corpus](https://www.cs.cmu.edu/~enron/) using **QLoRA** (4-bit quantized LoRA), optimized to run on **Google Colab's free tier** (T4 GPU, 16 GB VRAM).

### Multi-Task Capabilities

| Task | Input | Output |
|------|-------|--------|
| ✍️ **Email Composition** | Subject line + optional context | Full professional email |
| 💬 **Email Reply** | Received email | Contextual professional reply |

### Key Highlights

- 🔥 **QLoRA** fine-tuning with 4-bit NF4 quantization — fits in 16 GB VRAM
- 📊 **Comprehensive evaluation** with ROUGE, BLEU, BERTScore, and email-specific metrics
- 🚀 **Gradio demo app** deployable to HuggingFace Spaces
- 📓 **End-to-end Colab notebook** for one-click training

---

## 🧠 Why Llama 3.1 8B Instruct?

| Factor | Detail |
|--------|--------|
| **Performance** | Strongest general-purpose open model in the 7–8B parameter class |
| **Pre-training** | Already instruction-tuned → less training data & epochs needed |
| **Email aptitude** | Excellent at writing, summarization, and structured text generation |
| **LoRA stability** | Very stable for LoRA/QLoRA fine-tuning with predictable convergence |
| **Industry standard** | Widely used in production RAG systems and enterprise NLP pipelines |
| **Recruiter-friendly** | Demonstrates familiarity with industry-standard tooling |

---

## 📁 Project Structure

```
fine_tune/
│
├── 📂 data/
│   ├── raw/                        # Extract Enron dataset here
│   │   └── README.md               # Download & extraction instructions
│   ├── prepare_data.py             # Parse → clean → format → split pipeline
│   ├── raw.json                    # [generated] All formatted samples
│   ├── train.json                  # [generated] Training split (90%)
│   └── val.json                    # [generated] Validation split (10%)
│
├── 📂 training/
│   ├── config.py                   # Centralized hyperparameters & GPU detection
│   ├── train_lora.py               # QLoRA fine-tuning with SFTTrainer
│   └── merge_adapter.py            # Merge LoRA adapter → base model
│
├── 📂 evaluation/
│   ├── metrics.py                  # ROUGE, BLEU, BERTScore, format compliance
│   ├── evaluate.py                 # End-to-end evaluation runner
│   └── eval_results.ipynb          # Results visualization notebook
│
├── 📂 analysis/
│   └── error_analysis.ipynb        # Qualitative error analysis
│
├── 📂 app/
│   ├── app.py                      # Gradio demo (compose + reply)
│   └── requirements.txt            # App deployment dependencies
│
├── 📂 notebooks/
│   └── colab_training.ipynb        # Full Colab training pipeline
│
├── 📂 logs/                        # TensorBoard training logs
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/email-assistant-llama.git
cd email-assistant-llama
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download and extract the Enron email corpus:

```bash
# Download (~1.7 GB)
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

# Extract into data/raw/
tar -xzf enron_mail_20150507.tar.gz -C data/raw/

# Run the preparation pipeline
python data/prepare_data.py --maildir_path data/raw/maildir --max_samples 10000
```

This produces `train.json` (~9,000 samples) and `val.json` (~1,000 samples) with both compose and reply tasks.

### 4. Train on Google Colab

Open the Colab notebook and follow the step-by-step guide:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Or run locally (requires a GPU):

```bash
python training/train_lora.py
```

### 5. Evaluate

```bash
python evaluation/evaluate.py \
    --model_path outputs/final_adapter \
    --compare_base \
    --max_samples 100
```

### 6. Launch the Demo

```bash
python app/app.py
```

---

## 📊 Dataset

### Enron Email Corpus

The [Enron Email Dataset](https://www.cs.cmu.edu/~enron/) contains ~500,000 real corporate emails from ~150 Enron employees, made public during the 2001 federal investigation.

### Data Pipeline

```
Raw Enron Corpus (500K emails)
    │
    ▼
  Parse (email module)  ──→  Extract headers + body
    │
    ▼
  Clean  ──→  Remove forwards, quoted text, disclaimers, signatures
    │
    ▼
  Filter  ──→  Length limits, dedup by MD5, remove automated messages
    │
    ▼
  Multi-Task Format  ──→  Compose (subject → email) + Reply (email → reply)
    │
    ▼
  Split (90/10)  ──→  train.json + val.json
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | ~10,000 |
| Compose tasks | ~7,000–8,000 |
| Reply tasks | ~2,000–3,000 |
| Train split | ~9,000 (90%) |
| Val split | ~1,000 (10%) |
| Avg. email length | ~150 words |

---

## 🏋️ Training

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Quantization | 4-bit NF4 with double quantization |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | All attention + MLP projections |
| Batch size | 2 (× 4 gradient accumulation = 8 effective) |
| Learning rate | 2e-4 (cosine schedule) |
| Epochs | 3 |
| Max sequence length | 512 tokens |
| Optimizer | paged_adamw_32bit |
| Gradient checkpointing | ✅ Enabled |
| Compute dtype | float16 (T4 compatible) |

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | T4 (16 GB) | A100 (40 GB) |
| RAM | 12 GB | 16 GB |
| Disk | 20 GB | 50 GB |
| Training time | ~1–2 hours | ~30 min |

### Trainable Parameters

With LoRA rank 16 targeting all linear layers:

| Component | Parameters |
|-----------|-----------|
| Base model (frozen) | ~8B |
| LoRA adapters (trainable) | ~20M (~0.25%) |

---

## 📈 Evaluation

### Metrics

| Metric | What It Measures |
|--------|------------------|
| **ROUGE-1/2/L** | N-gram overlap with reference |
| **BLEU** | Precision of generated n-grams |
| **BERTScore** | Semantic similarity via embeddings |
| **Distinct-n** | Lexical diversity |
| **Format Compliance** | Email structure (greeting, body, closing) |

### Results

| Metric | Base Model | Fine-Tuned | Δ |
|--------|-----------|------------|---|
| ROUGE-1 | — | — | — |
| ROUGE-L | — | — | — |
| BLEU | — | — | — |
| BERTScore F1 | — | — | — |
| Format Compliance | — | — | — |

> *Results will be populated after training. Run `python evaluation/evaluate.py --compare_base` to fill this table.*

---

## 🌐 Deployment

### HuggingFace Spaces

1. **Merge the adapter** into the base model:
   ```bash
   python training/merge_adapter.py --adapter_path outputs/final_adapter
   ```

2. **Convert to GGUF** format (for CPU inference):
   ```bash
   python llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile model-f16.gguf --outtype f16
   ./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
   ```

3. **Upload the GGUF model** to a HuggingFace model repository

4. **Deploy the app** to HuggingFace Spaces:
   ```bash
   # Copy app files to a new Space
   cp app/app.py app/requirements.txt your-space-repo/
   # Set HF_MODEL_REPO environment variable in Space settings
   ```

### Local Inference

```bash
MODEL_PATH=./model-q4_k_m.gguf python app/app.py
```

---

## 📝 License

- **Code**: MIT License
- **Model**: [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/)
- **Dataset**: [Enron Email Dataset](https://www.cs.cmu.edu/~enron/) — Public domain (released by FERC)

## 🙏 Acknowledgments

- **[Meta AI](https://ai.meta.com/)** — Llama 3.1 model family
- **[Hugging Face](https://huggingface.co/)** — Transformers, PEFT, TRL, Spaces
- **[CMU](https://www.cs.cmu.edu/~enron/)** — Enron email dataset hosting
- **[Tim Dettmers](https://timdettmers.com/)** — QLoRA methodology & bitsandbytes

---

<p align="center">
  <i>Built as a portfolio project demonstrating production-grade LLM fine-tuning skills.</i>
</p>
