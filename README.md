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


## SparseGPT — One-Shot Pruning for LLMs

[SparseGPT](https://arxiv.org/abs/2301.00774) (Frantar & Alistarh, ICML 2023) shows that GPT-style models can be pruned **50–60% in a single pass**, *without retraining*, while preserving perplexity.  
The method reduces each layer’s pruning + reconstruction to fast, approximate sparse regression solved with second-order information from a small calibration set.

---


# SparseGPT — Mathematics & Method (GitHub-safe)

This section explains **SparseGPT** with equations that **render 100% reliably on GitHub** by embedding them as SVG images.  
You can keep this as your README section with guaranteed formatting (no GitHub math settings required).

---

## Layer Setup
For a layer with weight matrix $W \in \mathbb{{R}}^{{d_{{\text{{row}}}}\times d_{{\text{{col}}}}}}$ and calibration activations 
$X \in \mathbb{{R}}^{{d_{{\text{{col}}}}\times n}}$, define the (damped) empirical Hessian:

![Hessian]({eq_H_url})

---

## Pruning Objective (Eq. 1)
We seek a binary mask $M$ (1 = keep, 0 = prune) and reconstructed weights $W^c$ to minimize the output error:

![Objective]({eq_obj_url})

---

## Optimal Reconstruction with Fixed Mask (Eq. 2)
For each row $w_i$ with active indices $M_i$ (kept weights), the optimal reconstruction is:

![Row recon]({eq_row_url})

---

## OBS Error for Single Weight Removal (Eq. 3)
Removing weight $w_m$ yields the optimal local update and error:

![OBS]({eq_obs_url})

This error $\varepsilon_m$ is used to rank prune candidates.

---

## SparseGPT Algorithm (per layer)

- **Column-wise pruning** with a shared inverse-Hessian sequence.  
- **Block-wise selection**: process in blocks of size $B_s$ (typically 128).  
- **Scoring**: prune based on the OBS-inspired score below (lower is cheaper to prune).  
- **Lazy OBS updates**: compensate only **future** columns → big speedup in practice.

Score used in selection:

![Score]({eq_score_url})

**Complexity:** The reuse of the inverse-Hessian sequence reduces cost from $O(d_{{\text{{hidden}}}}^4)$ to $O(d_{{\text{{hidden}}}}^3)$.

---

## Extensions

**Weight Freezing View (Eq. 6):**

![Freeze]({eq_freeze_url})

**Joint Pruning + Quantization (Eq. 7):**

![Joint]({eq_joint_url})

**Semi-structured $n{:}m$ sparsity:** set group size $m$ (choose $B_s=m$), prune the $n$ lowest-scoring weights per group — supports **2:4**, **4:8**, etc.

---

## Practical Tips
- **Calibration set**: 64–128 sequences × 2048 tokens is typically enough.  
- **Dampening** $\lambda$: ≈ 1% of the average diagonal of $H$.  
- **Block size** $B_s$: around 128.

---

## Reference
Frantar, E. & Alistarh, D. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. ICML 2023. arXiv:2301.00774
