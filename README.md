# BERT Sentiment & Emotion Classifier

A Streamlit web app that analyzes movie reviews using a fine-tuned **BERT-base** model. It predicts the top emotions from the [GoEmotions](https://huggingface.co/datasets/go_emotions) dataset and maps them to **Positive**, **Negative**, or **Neutral** sentiment.

## Features

- **28 emotion labels** from GoEmotions (e.g. joy, anger, sadness, curiosity)
- **Sentiment derivation** — emotions are grouped into Positive, Negative, or Neutral
- **Top-3 predictions** with confidence bars
- **Streamlit UI** — paste a review and click Analyze

## Demo

Run locally:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`), enter a movie review, and click **Analyze Review**.

## Project structure

```
bert-sentiment-classifier/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes model weights from Git
├── model/              # Local weights (not pushed to GitHub)
│   ├── best_model.pt
│   └── emotion_encoder.pkl
└── README.md
```

Model weights are stored in `model/` for local use but are **not** committed to this repository. At runtime, the app downloads them from [Hugging Face](https://huggingface.co/sakshipawar2604/bert-sentiment-classifier).

## Model

| Component | Details |
|-----------|---------|
| Base model | [bert-base-uncased](https://huggingface.co/bert-base-uncased) |
| Task | Multi-class emotion classification |
| Training data | [GoEmotions](https://huggingface.co/datasets/go_emotions) |
| Architecture | BERT pooler → dropout (0.3) → linear classifier |
| Weights | Hosted on [sakshipawar2604/bert-sentiment-classifier](https://huggingface.co/sakshipawar2604/bert-sentiment-classifier) |

**Files on Hugging Face**

- `best_model.pt` — fine-tuned classifier weights
- `emotion_encoder.pkl` — label encoder for GoEmotions classes

## Installation

**Requirements:** Python 3.9+ recommended

1. Clone the repository:

   ```bash
   git clone https://github.com/sakshipawar2604/bert-sentiment-classifier.git
   cd bert-sentiment-classifier
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

On first run, PyTorch, the tokenizer, and model weights are downloaded from Hugging Face. Ensure you have an internet connection.

## Local model files (optional)

If you have trained weights locally, place them in `model/`:

```
model/best_model.pt
model/emotion_encoder.pkl
```

These paths are listed in `.gitignore` and will not be pushed to GitHub. The current `app.py` loads weights from Hugging Face Hub; to use local files instead, update the `hf_hub_download` paths in `load_encoder()` and `load_model()`.

## How it works

1. User enters a movie review in the text area.
2. Text is tokenized with `bert-base-uncased` (max length 128).
3. The fine-tuned BERT head outputs probabilities over 28 emotions.
4. The top emotion is mapped to sentiment via a fixed emotion → sentiment dictionary.
5. The UI shows predicted emotion, sentiment, and the top 3 emotion probabilities.

## Dependencies

See [`requirements.txt`](requirements.txt):

- `streamlit` — web UI
- `torch` — inference
- `transformers` — BERT tokenizer and backbone
- `huggingface_hub` — download model artifacts
- `pandas` — data utilities

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Sakshi Pawar** — [@sakshipawar2604](https://github.com/sakshipawar2604)
