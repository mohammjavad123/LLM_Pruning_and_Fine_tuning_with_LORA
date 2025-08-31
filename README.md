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

# SparseGPT Mathematics & Method

SparseGPT (Frantar & Alistarh, ICML 2023) introduces a one-shot pruning approach for large language models. 
It prunes **50–60% of weights** in a single pass, without retraining, while preserving perplexity. 
The method uses second-order information from a calibration set to decide which weights to prune and how to optimally compensate.

---

## Layer Setup
For a weight matrix $W \in \mathbb{R}^{d_{\text{row}} \times d_{\text{col}}}$ and calibration activations 
$X \in \mathbb{R}^{d_{\text{col}} \times n}$, define the (damped) empirical Hessian:

$$
H = X X^\top + \lambda I, 
\qquad 
H^{-1} = (X X^\top + \lambda I)^{-1}.
$$

---

## Pruning Objective (Eq. 1)
The goal is to find a binary mask $M$ (1 = keep, 0 = prune) and reconstructed weights $W^c$:

$$
\min_{M,\,W^c} \;\; \| W X - (M \odot W^c) X \|_2^2 .
$$

---

## Optimal Reconstruction with Fixed Mask (Eq. 2)
If $M$ is fixed, then for each row $w_i$ with active indices $M_i$:

$$
(w_i)_{M_i} 
= (X_{M_i} X_{M_i}^\top)^{-1} \, X_{M_i} \, \big((w_i)_{M_i} X_{M_i}\big)^\top .
$$

---

## OBS Error for Single Weight Removal (Eq. 3)
If we prune weight $w_m$, the optimal local update and associated error are:

$$
\delta_m = - \frac{w_m}{[H^{-1}]_{mm}} \, H^{-1}_{:,m}, 
\qquad 
\varepsilon_m = \frac{w_m^2}{[H^{-1}]_{mm}} .
$$

This score $\varepsilon_m$ is used to decide which weights are cheapest to remove.

---

## SparseGPT Algorithm
- **Column-wise pruning:** process weights column by column, reusing the same inverse Hessian sequence.  
- **Block-wise selection:** group columns into blocks of size $B_s$ (typically 128).  
- **Scoring:** prune weights with lowest ratio $w_c^2 / [H^{-1}]_{cc}$.  
- **Lazy OBS updates:** compensate only future columns → reduces cost from $O(d_{\text{hidden}}^4)$ to $O(d_{\text{hidden}}^3)$.

---

## Extensions

**Weight Freezing (Eq. 6):**

$$
\text{compress}(w_j)_i =
\begin{cases}
0 & j \notin M_i, \\
w^j_i & \text{otherwise}.
\end{cases}
$$

**Joint Pruning + Quantization (Eq. 7):**

$$
E_{:,j-i} \leftarrow 
\frac{W_{:,j} - M_{:,j} \odot \mathrm{quant}(W_{:,j})}{[H^{-1}]_{jj}} .
$$

**Structured $n{:}m$ sparsity:** set block size $B_s = m$, prune the $n$ lowest-scoring weights per group (supports 2:4, 4:8, etc.).

---

### 📊 Results (from paper)

- **OPT-175B / BLOOM-176B:** ~4h on a single A100-80GB  
- Achieves **50–60% sparsity** with minimal perplexity increase  
- Compatible with **4-bit quantization** in one unified pass  


1. Load a HuggingFace model (e.g. OPT).  
2. Collect calibration data (default: **C4** subset).  
3. Run SparseGPT for unstructured or \(n{:}m\) pruning.  
4. Measure perplexity on WikiText2 / PTB / C4.  

