# GrowCode
GrowCode – a small Phi-3–based DSA coding helper, fine-tuned with LoRA to explain solutions, find bugs, give hints, and optimize code.
# 🌱🚀 GrowCode – DSA Coding Helper (Phi-3 + LoRA)

GrowCode is a small coding assistant fine-tuned on top of **Phi-3 mini (microsoft/Phi-3-mini-4k-instruct)** to help with **DSA / LeetCode-style problems**.

It can:
- ✏️ **Explain solutions** in simple, step-by-step language  
- 🐞 **Find bugs and explain** what’s wrong in your code  
- 💡 **Give hints** without spoiling the full solution  
- 🧹 **Suggest optimizations** for time/space complexity  

Everything runs **locally** via a minimal **Gradio** UI – no external API calls required once the model is downloaded.

---

## ✨ Features

- **Fine-tuned Phi-3 mini with LoRA** using `trl` + `peft`
- **Instruction-style dataset** stored as `train.jsonl` with fields:
  - `instruction` – what kind of help (explain, bug, hint, optimize)
  - `input` – problem statement + code
  - `output` – ideal tutor-style response  
- Custom prompt format with `<system> / <user> / <assistant>` tags
- Simple **Gradio interface** with four modes:
  - `Explain the solution`
  - `Find the bug and explain`
  - `Give hint and solution`
  - `Optimize the code`
- Runs on **Apple Silicon (M-series)** with `mps` (GPU) or CPU fallback

---

## 🧠 Tech Stack

- **Base model:** `microsoft/Phi-3-mini-4k-instruct`
- **Fine-tuning:**  
  - [Transformers](https://github.com/huggingface/transformers)  
  - [TRL](https://github.com/huggingface/trl) (`SFT` style)  
  - [PEFT](https://github.com/huggingface/peft) with **LoRA**
  - `datasets` for JSONL loading + splits
- **Serving / UI:**  
  - [Gradio](https://github.com/gradio-app/gradio)  
- **Hardware:**  
  - Trained on Google Colab GPU  
  - Inference on local Mac (M-series) using `torch.backends.mps`

---

## 🏗️ Project Structure (example)

```bash
.
├── app.py                     # Gradio app (GrowCode UI)
├── Phi_coder.py               # Local inference script (Phi-3 + UI)
├── train.jsonl                # Instruction dataset for fine-tuning
├── notebooks/
│   └── fine_tune_phi3.ipynb   # Colab notebook for LoRA training
└── phi3_coder_lora_adapter/   # Saved LoRA adapter + tokenizer

# 1. Create & activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Install deps
pip install -r requirements.txt
# or manually: transformers peft trl datasets gradio torch

# 3. Run GrowCode
python3 Phi_coder.py
Run here : http://127.0.0.1:7860
