# 🧠 NLP Similarity + Writing Psychology Analyzer

> NLP Lab Quiz — Shifa Tameer-e-Millat University

## 📌 App Purpose

A dual-purpose Streamlit app that:
1. **Text/Word Similarity** — uses a free pretrained sentence-transformer model to compute semantic similarity between any words or sentences, with Bar Chart, Heatmap, and PCA visualizations.
2. **Writing Psychology Analyzer** — accepts a `.docx` or `.txt` upload (or pasted text) and analyzes the writer's cognitive thinking style across 5 psychological profiles using keyword density + semantic similarity.

## 🤖 Model Used

**`sentence-transformers/all-MiniLM-L6-v2`**
- Free, open-source pretrained model from HuggingFace
- No training performed
- No manual preprocessing applied
- Produces 384-dimensional sentence embeddings

## 🧬 Psychology Profiles Detected

| Profile | Description |
|---|---|
| Critical Thinker | Challenges assumptions, evaluates evidence |
| Analytical Thinker | Systematic, data-driven, structured |
| Creative Thinker | Novel ideas, explores possibilities |
| Reflective Thinker | Personal values, inner experience |
| Pragmatic Thinker | Real-world solutions, action-oriented |

## 📊 Graphs Included

| Graph | Purpose |
|---|---|
| Bar Chart | Top similar items by cosine score |
| Heatmap | Pairwise similarity matrix |
| PCA 2D Plot | Embedding clusters in reduced space |
| Psychology Bar | Composite profile scores |
| Radar Chart | Keyword density across profiles |
| Writing Metrics | Sentence length, vocabulary richness, question count |

## 🧠 Paul's Critical Thinking Standards

Applied inline in the app for both tabs:
**Clarity · Accuracy · Precision · Relevance · Logic · Significance · Fairness**

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Streamlit Deployed App

[🔗 Add your Streamlit Cloud link here after deployment]

## 📁 Repository Structure

```
├── app.py             # Main Streamlit application
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## 📸 Screenshots

_Add screenshots of the running app here after deployment_
