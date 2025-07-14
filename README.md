# Auto-tagging-support-llm
# Auto-tagging-supports-llm
# 🎫 Auto-Tagging Support Tickets with LLM

This project automatically tags support tickets into categories using multiple approaches: Zero-shot, Few-shot, and Fine-tuned models. It also shows the top 3 most probable tags with a beautiful visual interface.

## 🚀 Features

✅ Zero-shot classification using local TinyLlama model  
✅ Few-shot classification with custom examples  
✅ Fine-tuned model classification (Hugging Face-based)  
✅ Top 3 tags display with bar charts (Plotly)  
✅ Dark-themed Streamlit web app interface  
✅ Flexible interactive examples in the sidebar

---

## 🏷️ Tags

- Billing
- Login Issue
- Technical Bug
- Feature Request
- Refund Request

---
## 🔥 Download fine-tuned model

The large fine-tuned model file (`model.safetensors`) is stored separately to keep this repo lightweight.

### 👉 [Download model from Google Drive](https://drive.google.com/drive/folders/1ut6btIfZpwwIORnxkzcR-cz7JkArKB-3?usp=drive_link)

After downloading, place it inside:



## 🛠️ Installation

### Clone this repo

```bash
git clone https://github.com/Mohammed-Umair30/Auto-tagging-supports-llm.git
cd auto-tagging-support-tickets

Create virtual environment (recommended)

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate
Install requirements

pip install -r requirements.txt
⚡ Usage

Run the Streamlit app

streamlit run app.py

Choose your method
Zero-shot: Classifies using LLM without examples.

Few-shot: Classifies using few example tickets (configurable in sidebar).

Fine-tuned: Uses your pre-trained local model (e.g., fine-tuned on Google Colab).

📄 Dataset

The dataset used for fine-tuning can be based on Twitter Customer Support Dataset (TWCS).

You can customize or expand to include your real support tickets.

