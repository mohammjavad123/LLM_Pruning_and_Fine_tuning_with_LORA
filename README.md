# Fine-Tuning Large Language Models with LoRA in Python

This repository demonstrates how to **efficiently fine-tune large language models (LLMs)** using **LoRA (Low-Rank Adaptation)**, significantly reducing memory requirements and training time compared to full model fine-tuning.  
The project includes both **theoretical explanations** and **practical implementations** using Hugging Face's `transformers` and `PEFT` libraries, with 4-bit quantization via `bitsandbytes` for resource efficiency.

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
  - Training completed on a single consumer GPU (RTX 3060 Ti, 8GB VRAM) using 4-bit quantization.
  - Memory footprint was reduced dramatically compared to full fine-tuning.

---

## 📂 Repository Structure

```
.
├── notebooks/
│   ├── lora_theory_and_gsm8k.ipynb     # Theory + GSM-8K fine-tuning
│   ├── lora_custom_dataset.ipynb       # Fine-tuning on custom dataset
│   └── lora_evaluation.ipynb           # Evaluation & perplexity testing
├── data/
│   ├── custom_dataset.jsonl            # Example custom dataset
│   └── gsm8k/                          # GSM-8K subset used for training
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/lora-finetuning.git
cd lora-finetuning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch`
- `transformers`
- `datasets`
- `peft`
- `bitsandbytes`

### 3. Prepare Data
- Place your dataset in `data/` (supports JSONL, CSV, or Hugging Face datasets).
- Update dataset paths in the notebooks.

### 4. Run Fine-Tuning
Open and execute the provided notebooks:
- **`lora_theory_and_gsm8k.ipynb`** – Learn LoRA theory and fine-tune on GSM-8K.
- **`lora_custom_dataset.ipynb`** – Fine-tune on your own dataset.

---

## 🧪 Evaluation

We evaluate the model using:
- **Perplexity**: Lower values indicate better adaptation.
- **Qualitative checks**: Compare reasoning and answer style between base and tuned models.

---

## 📦 Model Saving & Loading

**Save Adapter:**
```python
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")
```

**Load Adapter:**
```python
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, "lora_adapter")
```

---

## 📈 Key Takeaways

- LoRA allows **task-specific fine-tuning** without retraining the full model.
- **4-bit quantization** makes fine-tuning feasible on low-VRAM GPUs.
- Adapters are **lightweight and reusable**, enabling efficient multi-task workflows.

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing
Pull requests are welcome!  
For major changes, please open an issue first to discuss what you’d like to modify.

---

## ⭐ Acknowledgments
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- Original LoRA paper: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
