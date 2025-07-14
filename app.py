import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

# ---- Setup ----
tags = ["Billing", "Login Issue", "Technical Bug", "Feature Request", "Refund Request"]

llm = ChatOllama(
    model="tinyllama:latest",
    model_kwargs={"num_predict": 300}
)

model_path = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
softmax = nn.Softmax(dim=1)

# ---- Helper functions ----
def parse_top3_response(raw_text, tags):
    # Convert raw text to lowercase for robust partial matching
    raw_text_lower = raw_text.lower()
    valid_parts = []

    for tag in tags:
        if tag.lower() in raw_text_lower:
            valid_parts.append(tag)

    # In case LLM still returns messy text, extract possible tag words
    if not valid_parts:
        # Try to split raw text and find any individual tag words
        raw_parts = [x.strip() for x in raw_text.replace("\n", ",").split(",")]
        for part in raw_parts:
            for tag in tags:
                if tag.lower() in part.lower() and tag not in valid_parts:
                    valid_parts.append(tag)

    while len(valid_parts) < 3:
        valid_parts.append("Other")

    return valid_parts[:3]


def zero_shot_classify(ticket_text, tags, llm):
    prompt = f"Classify this support ticket into one of these categories: {', '.join(tags)}.\n\nTicket: \"{ticket_text}\"\nProvide only the category name."
    response = llm.invoke(prompt)
    return response.content.strip()

def zero_shot_top3(ticket_text, tags, llm):
    prompt = (
        f"Given this support ticket: \"{ticket_text}\", choose and rank exactly 3 categories from this list: {', '.join(tags)}.\n"
        f"Return only the category names, separated by commas. No explanation, no extra words."
    )
    response = llm.invoke(prompt)
    top_tags = parse_top3_response(response.content.strip(), tags)
    return top_tags


def few_shot_classify(ticket_text, tags, llm, examples):
    example_text = ""
    for ex in examples:
        example_text += f"Ticket: \"{ex['text']}\"\nCategory: {ex['tag']}\n\n"
    prompt = (
        f"{example_text}"
        f"Ticket: \"{ticket_text}\"\n"
        f"Provide only the category name from this list (no explanation): {', '.join(tags)}.\n"
        f"Category:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def few_shot_top3(ticket_text, tags, llm, examples):
    example_text = ""
    for ex in examples:
        example_text += f"Ticket: \"{ex['text']}\"\nCategory: {ex['tag']}\n\n"
    prompt = (
        f"{example_text}"
        f"Ticket: \"{ticket_text}\"\n"
        f"Rank and return exactly 3 categories from this list: {', '.join(tags)}.\n"
        f"Return only the category names, separated by commas. No explanation, no extra words."
    )
    response = llm.invoke(prompt)
    top_tags = parse_top3_response(response.content.strip(), tags)
    return top_tags


def fine_tuned_classify(ticket_text, model, tokenizer, labels):
    inputs = tokenizer(ticket_text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits)
    top_probs, top_indices = torch.topk(probs, k=3)
    top_probs = top_probs[0].tolist()
    top_indices = top_indices[0].tolist()
    top_tags = [(labels[i], round(p * 100, 2)) for i, p in zip(top_indices, top_probs)]
    main_tag = labels[top_indices[0]]
    return main_tag, top_tags, probs[0].tolist()

def plot_top3(tags_ranked):
    assumed_scores = [70, 20, 10]  # Mock confidences for LLM-based
    df = pd.DataFrame({
        "Tag": tags_ranked,
        "Confidence (%)": assumed_scores
    })
    fig = px.bar(df, x="Confidence (%)", y="Tag", orientation="h", color="Confidence (%)", color_continuous_scale="blues")
    return fig

# ---- Streamlit UI ----
st.set_page_config(page_title="🎫 Auto-Tagging Support Tickets", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
        .stApp {background-color: #111;}
        h1, h2, h3, h4, h5, h6, p, label, div, span, textarea, input, select, button {
            color: #eee !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎫 Auto-Tagging Support Tickets")
st.markdown("Use local LLM or fine-tuned model to automatically tag support tickets. Choose a method and optionally see top 3 tags.")

ticket_text = st.text_area("Enter your support ticket text here:", height=150)

method = st.sidebar.selectbox("Choose classification method", ["Zero-shot", "Few-shot", "Fine-tuned"])
show_top3 = st.sidebar.checkbox("Show Top 3 Tags", value=True)

if method == "Few-shot":
    st.sidebar.markdown("### Few-shot Examples")
    ex1 = st.sidebar.text_input("Example 1 Text", "I want a refund for my last payment.")
    ex1_tag = st.sidebar.selectbox("Example 1 Tag", tags, index=4)
    ex2 = st.sidebar.text_input("Example 2 Text", "My app crashes every time I open it.")
    ex2_tag = st.sidebar.selectbox("Example 2 Tag", tags, index=2)
    ex3 = st.sidebar.text_input("Example 3 Text", "I can't log in even after resetting my password.")
    ex3_tag = st.sidebar.selectbox("Example 3 Tag", tags, index=1)

if st.button("🚀 Classify Ticket"):
    if not ticket_text.strip():
        st.warning("Please enter some ticket text.")
    else:
        if method == "Zero-shot":
            if show_top3:
                top_tags = zero_shot_top3(ticket_text, tags, llm)
                st.success("✅ **Top 3 Tags (Zero-shot):**")
                for i, tag in enumerate(top_tags, 1):
                    st.markdown(f"{i}. **{tag}**")
                fig = plot_top3(top_tags)
                st.plotly_chart(fig, use_container_width=True)
            else:
                prediction = zero_shot_classify(ticket_text, tags, llm)
                st.success("✅ **Predicted Tag (Zero-shot):**")
                st.markdown(f"<h3 style='color: lightgreen;'>{prediction}</h3>", unsafe_allow_html=True)

        elif method == "Few-shot":
            examples = [
                {"text": ex1, "tag": ex1_tag},
                {"text": ex2, "tag": ex2_tag},
                {"text": ex3, "tag": ex3_tag},
            ]
            if show_top3:
                top_tags = few_shot_top3(ticket_text, tags, llm, examples)
                st.success("✅ **Top 3 Tags (Few-shot):**")
                for i, tag in enumerate(top_tags, 1):
                    st.markdown(f"{i}. **{tag}**")
                fig = plot_top3(top_tags)
                st.plotly_chart(fig, use_container_width=True)
            else:
                prediction = few_shot_classify(ticket_text, tags, llm, examples)
                st.success("✅ **Predicted Tag (Few-shot):**")
                st.markdown(f"<h3 style='color: lightgreen;'>{prediction}</h3>", unsafe_allow_html=True)

        else:  # Fine-tuned
            main_tag, top_tags_probs, all_probs = fine_tuned_classify(ticket_text, model, tokenizer, tags)
            st.success("✅ **Predicted Tag (Fine-tuned):**")
            st.markdown(f"<h3 style='color: lightgreen;'>{main_tag}</h3>", unsafe_allow_html=True)
            if show_top3:
                st.subheader("🔝 Top 3 Tags with Confidence")
                for tag, prob in top_tags_probs:
                    st.markdown(f"- **{tag}**: {prob}%")

                df_probs = pd.DataFrame({"Tag": tags, "Confidence (%)": [round(p * 100, 2) for p in all_probs]})
                fig = px.bar(df_probs.sort_values("Confidence (%)", ascending=True),
                             x="Confidence (%)", y="Tag", orientation="h", color="Confidence (%)", color_continuous_scale="greens")
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using LangChain, Hugging Face Transformers, and Streamlit.")
