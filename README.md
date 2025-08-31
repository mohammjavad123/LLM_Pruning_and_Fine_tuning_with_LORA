# Fine-Tuning Large Language Models with LoRA in Python

This repository demonstrates how to **efficiently fine-tune large language models (LLMs)** using **LoRA (Low-Rank Adaptation)**, significantly reducing memory requirements and training time compared to full model fine-tuning.  
The project includes both **theoretical explanations** and **practical implementations** using Hugging Face's `transformers` and `PEFT` libraries, with 4-bit quantization via `bitsandbytes` for resource efficiency.
In addition to LoRA fine-tuning, we integrated SparseGPT pruning to reduce the base model size and speed up inference. SparseGPT applies activation-aware, block-wise pruning to remove redundant weights while maintaining performance. We evaluated the base model, the SparseGPT-pruned model on and summarization  tasks. Evaluation included perplexity for reasoning and ROUGE scores for summarization, allowing us to measure the trade-offs between efficiency gains and task performance across different sparsity levels.

---

## 📌 Overview

**Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning method where instead of updating the entire set of model weights, we train two small low-rank matrices that represent the weight updates (ΔW). These updates can be added to the base model during inference, enabling:

- **Faster training** due to fewer parameters being updated.
- **Lower GPU memory usage**, making fine-tuning feasible on consumer hardware.
- **Reusable adapters** for different tasks, without retraining the full model.

This repository covers:
- Theoretical background on LoRA and its matrix decomposition approach.
- Fine-tuning an open-source model on both a public dataset and a custom dataset.
- Evaluation using **perplexity** and qualitative output comparison.
- Saving, loading, and swapping LoRA adapters for multiple tasks.

---

## 🎯 Project Aim

The goal is to show that **LoRA can adapt LLMs to specific tasks** with minimal compute resources while maintaining strong performance and generalization.

---

## 🛠 What We Did

1. **Theoretical Foundation**  
   - Explained LoRA’s low-rank decomposition concept and why it works for large models.
   - Discussed rank selection, scaling factors, and targeted modules for efficient adaptation.

2. **Implementation in Python**  
   - Used Hugging Face `transformers`, `datasets`, and `PEFT` for model loading and fine-tuning.
   - Integrated `bitsandbytes` 4-bit quantization for reduced VRAM usage.
   - Set up LoRA configuration (rank, target layers, dropout, scaling).

3. **Fine-Tuning**  
   - **Public Dataset**: GSM-8K math reasoning dataset for proof-of-concept fine-tuning.  
   - **Custom Dataset**: A domain-specific dataset requiring adaptation to task-specific patterns and reasoning styles.

4. **Evaluation**  
   - Computed **perplexity** to quantitatively measure model improvement.
   - Compared outputs of base and fine-tuned models to assess style and reasoning changes.
   - Tested on both training and unseen test data for generalization.

5. **Model Deployment**  
   - Saved LoRA adapters and tokenizer for reuse.
   - Demonstrated loading and merging adapters into the base model for quick task switching.

---

## 📊 Results

- **Perplexity Reduction**: Fine-tuned models showed significantly lower perplexity than base models on both training and unseen test data.
- **Qualitative Improvements**:  
  - GSM-8K fine-tuned models adapted their reasoning style and output formatting to match training examples.
  - On the custom dataset, the model successfully learned domain-specific patterns and generalized to unseen cases.
- **Efficiency Gains**:  
  - Memory footprint was reduced dramatically compared to full fine-tuning.


---



