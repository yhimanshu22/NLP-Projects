# 🚀 NLP Projects: Advanced Language Modeling Pipeline

This repository hosts a collection of production-ready NLP projects focused on **Text Generation**, **Machine Translation**, and **Sentiment Analysis**. All projects are implemented using a standardized **7-Phase Modular Pipeline** for maximum reproducibility and scalability.

## 🏗️ The 7-Phase Pipeline Structure
Every project in this repository follows a rigorous engineering structure:
1.  **Environment Setup**: Dependency management and hardware verification.
2.  **Configuration**: Centralized hyperparameters and model IDs.
3.  **Data Acquisition**: Automated dataset loading from Hugging Face.
4.  **Preprocessing**: Advanced tokenization and data formatting.
5.  **Model Preparation**: 4-bit quantization (bitsandbytes) and LoRA (PEFT).
6.  **Training**: Optimized fine-tuning using `SFTTrainer` or `Trainer` API.
7.  **Inference**: Real-time testing and performance validation.

---

## 📂 Project Overview

### 1. Hinglish Text Generation (Llama-2)
Fine-tuning the TinyLlama model for **Hinglish** (Hindi + English) conversational text generation.
-   **Local Notebook**: [`llama2_hinglish_gen.ipynb`](./llama2_hinglish_gen.ipynb)
-   **Colab Reference**: [Text Generation in Hinglish](https://colab.research.google.com/drive/1_M-6kwsGlP2mblV5tBazc6lKuYgQ8E9R)
-   **Key Features**: Parameter-efficient fine-tuning (LoRA), 4-bit quantization, and Alpaca-style prompt engineering.

### 2. French-to-English Translation
End-to-end translation pipeline optimized for high-quality French-English bilingual tasks.
-   **Local Notebook**: [`llama2_en_fr_pipeline.ipynb`](./llama2_en_fr_pipeline.ipynb)
-   **Colab Reference**: [French-to-English Translation](https://colab.research.google.com/drive/1phODqVfpFwcQHpG9gondIUDtTM6sT4_L)

### 3. English-to-Bengali Translation
Fine-tuning Llama-2 (TinyLlama) for accurate English-to-Bengali translation.
-   **Local Notebook**: [`llama2_bn_pipeline.ipynb`](./llama2_bn_pipeline.ipynb)
-   **Colab Reference**: [English-to-Bengali Translation](https://colab.research.google.com/drive/1ggFd6sYfS5G9iQbGkUPHwfJCbEm87R3A)

### 4. English-to-Hindi Translation
Efficient English-to-Hindi translation using a streamlined Llama-2 pipeline.
-   **Local Notebook**: [`llama2_en_hi_pipeline.ipynb`](./llama2_en_hi_pipeline.ipynb)
-   **Colab Reference**: [English-to-Hindi Translation](https://colab.research.google.com/drive/1B0gj0RpJ1uxW7CjEvooP5Bj1FGPX6HAa)

### 5. Indian Politics Sentiment Analysis
Multilingual sentiment classification for Indian political discourse using XLM-RoBERTa.
-   **Local Notebook**: [`politics_sentiment_analysis.ipynb`](./politics_sentiment_analysis.ipynb)
-   **Colab Reference**: [Indian Politics Sentiment Analysis](https://colab.research.google.com/drive/1sQtYp5oTfuP5L2mqnAytxEtWhy62UYLN)

---

## 🛠️ Setup & Requirements

```bash
# Clone the repository
git clone https://github.com/yhimanshu22/NLP-Projects
cd NLP-Projects

# Install core dependencies
pip install transformers trl datasets accelerate peft bitsandbytes
```

## 🚀 Usage
1.  Open any `.ipynb` notebook in VS Code or Google Colab.
2.  Ensure you have an NVIDIA GPU for the 4-bit quantization steps.
3.  Run the phases sequentially from Phase 1 to Phase 7.
