import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from huggingface_hub import hf_hub_download
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Emotion & Sentiment Classifier", layout="centered")

# Load encoder from model repo
@st.cache_resource
def load_encoder():
    encoder_path = hf_hub_download(
        repo_id="sakshipawar2604/bert-sentiment-classifier",
        filename="emotion_encoder.pkl"
    )
    with open(encoder_path, "rb") as f:
        emotion_encoder = pickle.load(f)
    return emotion_encoder, list(emotion_encoder.classes_)

# Sentiment mapping
emotion_to_sentiment = {
    'admiration': 'Positive', 'amusement': 'Positive', 'approval': 'Positive',
    'gratitude': 'Positive', 'joy': 'Positive', 'love': 'Positive',
    'optimism': 'Positive', 'pride': 'Positive', 'relief': 'Positive',
    'anger': 'Negative', 'disgust': 'Negative', 'disappointment': 'Negative',
    'embarrassment': 'Negative', 'fear': 'Negative', 'grief': 'Negative',
    'nervousness': 'Negative', 'remorse': 'Negative', 'sadness': 'Negative',
    'confusion': 'Neutral', 'curiosity': 'Neutral', 'realization': 'Neutral',
    'surprise': 'Neutral', 'neutral': 'Neutral', 'desire': 'Neutral',
    'caring': 'Neutral', 'excitement': 'Positive'
}

def sentiment_color(sentiment):
    return {
        "Positive": "#1b4332",
        "Neutral":  "#1d3557",
        "Negative": "#641e16"
    }.get(sentiment, "#333")

# Model definition
class EmotionClassifier(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.emotion_classifier = nn.Linear(768, num_emotions)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        logits = self.emotion_classifier(dropped)
        return logits

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@st.cache_resource
def load_model(num_labels):
    model_path = hf_hub_download(
        repo_id="sakshipawar2604/bert-sentiment-classifier",
        filename="best_model.pt"
    )
    model = EmotionClassifier(num_emotions=num_labels)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Load encoder + model
emotion_encoder, goemotions_labels = load_encoder()
tokenizer = load_tokenizer()
model = load_model(num_labels=len(goemotions_labels))

# Predict top emotions
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top_indices = probs.argsort()[-3:][::-1]
        top_emotions = [(goemotions_labels[i], round(float(probs[i] * 100), 2)) for i in top_indices]
        return top_emotions, probs

def derive_sentiment(emotion):
    return emotion_to_sentiment.get(emotion, "Neutral")

# Custom styling
st.markdown("""
    <style>
    .stButton > button {
        background-color: #e63946;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #b83232;
        color: white;
    }
    .result-box {
        padding: 30px 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# UI Layout
st.markdown("<h1 style='text-align:center; color:white;'>Emotion & Sentiment Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze a movie review to identify the most likely emotional tone and sentiment.</p>", unsafe_allow_html=True)

user_input = st.text_area("Enter your movie review below:", height=150)

# Centered form-style analyze button
st.markdown("""
    <div style='text-align:center; margin-top: 1rem; margin-bottom: 2rem;'>
        <form action="" method="post">
            <button type="submit" name="analyze" style="
                background-color: #e63946;
                color: white;
                font-weight: 600;
                padding: 0.6rem 2rem;
                font-size: 1rem;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            ">🔍 Analyze Review</button>
        </form>
    </div>
""", unsafe_allow_html=True)

analyze = st.session_state.get("analyze", False)
if st.session_state.get("analyze_flag", False):
    analyze = True
    st.session_state.analyze_flag = False

if st.experimental_get_query_params().get("analyze") is not None:
    st.session_state.analyze_flag = True
    st.experimental_set_query_params()
    st.rerun()

if analyze and user_input.strip():
    top_emotions, probs = predict_emotions(user_input)
    final_emotion = top_emotions[0][0]
    confidence = top_emotions[0][1]
    sentiment = derive_sentiment(final_emotion)
    sent_color = sentiment_color(sentiment)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown(f"""
            <div class='result-box' style='background-color:#1c2959; text-align:center;'>
                <h4>Predicted Emotion</h4>
                <h2 style='font-size:2.4rem;'>{final_emotion.title()}</h2>
                <p>Confidence: <strong>{confidence:.2f}%</strong></p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='result-box' style='background-color:{sent_color}; text-align:center;'>
                <h4>Detected Sentiment</h4>
                <h2 style='font-size:2.4rem;'>{sentiment}</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Top 3 Emotion Probabilities")

    for i, (emo, conf) in enumerate(top_emotions, start=1):
        st.markdown(f"""
        <div style='display:flex; align-items:center; margin-bottom:8px;'>
            <span style='font-size:20px; margin-right:8px;'>✔️</span>
            <strong style='flex:1;'>{emo.title()}</strong>
            <div style='width:60%; margin:0 10px; background:#ddd; border-radius:8px;'>
                <div style='width:{conf}%; background:#3b82f6; height:12px; border-radius:8px;'></div>
            </div>
            <span>{conf:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align:center;'>Model::BERT-base • Trained on GoEmotions  |  Built with using Streamlit</p>", unsafe_allow_html=True)

elif analyze:
    st.warning("Please enter a review to analyze.")