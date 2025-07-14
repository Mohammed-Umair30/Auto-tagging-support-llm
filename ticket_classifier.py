import pandas as pd
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# TinyLlama local setup
llm = ChatOllama(
    model="tinyllama:latest",
    model_kwargs={"num_predict": 300}
)

tags = ["Billing", "Login Issue", "Technical Bug", "Feature Request", "Refund Request"]

def zero_shot_classify(ticket_text, tags, llm):
    prompt = f"Classify this support ticket into one of these categories: {', '.join(tags)}.\n\nTicket: \"{ticket_text}\".\nCategory:"
    response = llm.invoke(prompt)
    return response.content.strip()


def few_shot_classify(ticket_text, tags, llm, examples):
    example_text = ""
    for ex in examples:
        example_text += f"Ticket: \"{ex['text']}\"\nCategory: {ex['tag']}\n\n"
    
    prompt = (
        f"{example_text}"
        f"Now classify this ticket into one of these categories: {', '.join(tags)}.\n\n"
        f"Ticket: \"{ticket_text}\".\nCategory:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


# Load fine-tuned model
model_path = "fine_tuned_model"  # extracted downloaded folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def fine_tuned_classify(ticket_text, model, tokenizer, labels):
    inputs = tokenizer(ticket_text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return labels[predicted_label]

# Example test
ticket_text = "I cannot log into my account even after resetting the password."

# Zero-shot
tag_zero = zero_shot_classify(ticket_text, tags, llm)
print("Zero-shot predicted tag:", tag_zero)

# Few-shot
examples = [
    {"text": "I want a refund for my last payment.", "tag": "Refund Request"},
    {"text": "My app crashes every time I open it.", "tag": "Technical Bug"},
    {"text": "I can't log in even after resetting my password.", "tag": "Login Issue"},
]
tag_few = few_shot_classify(ticket_text, tags, llm, examples)
print("Few-shot predicted tag:", tag_few)

# Fine-tuned
tag_ft = fine_tuned_classify(ticket_text, model, tokenizer, tags)
print("Fine-tuned predicted tag:", tag_ft)
