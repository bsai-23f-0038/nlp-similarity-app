import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import io
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import docx

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Similarity + Writing Psychology",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
  body { background-color: #0d0d0d; }
  .block-container { padding-top: 1.5rem; }
  h1 { color: #c9f542; font-family: 'Courier New', monospace; }
  h2, h3 { color: #e0e0e0; }
  .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
  .metric-card {
    background: #1a1a2e;
    border-left: 4px solid #c9f542;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
  }
  .psych-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 15px;
    margin: 4px;
  }
  .tag-pill {
    background: #1e3a5f;
    color: #9ecfff;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 13px;
    margin: 2px;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ─── Helpers ───────────────────────────────────────────────────────────────────
def embed(texts):
    return model.encode(texts, convert_to_numpy=True)

def cosine_sim(a, b):
    return float(cosine_similarity([a], [b])[0][0])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")

# ─── Psychology Analysis Engine ────────────────────────────────────────────────
PSYCH_PROFILES = {
    "Critical Thinker": {
        "keywords": ["however", "although", "despite", "yet", "but", "on the other hand",
                     "contradicts", "evaluate", "analyze", "critique", "flawed", "evidence",
                     "therefore", "conclude", "assumption", "biased", "validity", "argue",
                     "question", "doubt", "limitation"],
        "color": "#c9f542",
        "description": "Challenges ideas, questions assumptions, and evaluates evidence before accepting conclusions."
    },
    "Analytical Thinker": {
        "keywords": ["because", "thus", "hence", "specifically", "precisely", "data",
                     "result", "measure", "calculate", "pattern", "structure", "process",
                     "method", "systematic", "formula", "define", "classify", "category",
                     "step", "sequence"],
        "color": "#42c8f5",
        "description": "Breaks down complex problems into structured parts and relies on logic and data."
    },
    "Creative Thinker": {
        "keywords": ["imagine", "what if", "could", "might", "innovative", "new",
                     "unique", "creative", "design", "idea", "explore", "possibility",
                     "vision", "novel", "brainstorm", "alternatively", "differently",
                     "original", "invent", "transform"],
        "color": "#f542b9",
        "description": "Generates novel ideas, explores possibilities, and thinks beyond conventions."
    },
    "Reflective Thinker": {
        "keywords": ["feel", "believe", "think", "personally", "in my view", "i learned",
                     "realized", "understand", "experience", "meaning", "value", "purpose",
                     "growth", "lesson", "reflect", "perspective", "insight", "myself",
                     "journey", "opinion"],
        "color": "#f5a442",
        "description": "Turns inward to examine personal values, beliefs, and experiences to make meaning."
    },
    "Pragmatic Thinker": {
        "keywords": ["practical", "implement", "solution", "work", "apply", "use",
                     "effective", "efficient", "result", "action", "goal", "plan",
                     "achieve", "execute", "real-world", "feasible", "deliver",
                     "deploy", "build", "produce"],
        "color": "#42f5a4",
        "description": "Focuses on workable solutions and real-world outcomes over theoretical ideals."
    },
}

def analyze_psychology(text):
    if not text or len(text.strip()) < 50:
        return None

    text_lower = text.lower()
    words_total = len(text_lower.split())

    # Count keyword hits per profile
    scores = {}
    matched = {}
    for profile, data in PSYCH_PROFILES.items():
        hits = []
        for kw in data["keywords"]:
            count = text_lower.count(kw)
            if count > 0:
                hits.append((kw, count))
        scores[profile] = sum(c for _, c in hits) / max(words_total, 1) * 100
        matched[profile] = hits

    # Sentence complexity
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    # Vocabulary richness
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    unique_ratio = len(set(words)) / max(len(words), 1)

    # Question marks = curious/critical
    q_count = text.count("?")

    # Passive voice proxy
    passive_hits = len(re.findall(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', text_lower))

    # Embedding-based similarity to each profile description
    profile_descs = [d["description"] for d in PSYCH_PROFILES.values()]
    profile_names = list(PSYCH_PROFILES.keys())
    text_emb = embed([text[:1000]])[0]
    desc_embs = embed(profile_descs)
    semantic_scores = {
        name: float(cosine_similarity([text_emb], [desc_embs[i]])[0][0]) * 100
        for i, name in enumerate(profile_names)
    }

    # Blend keyword + semantic scores
    final_scores = {}
    for p in PSYCH_PROFILES:
        final_scores[p] = (scores[p] * 0.55) + (semantic_scores[p] * 0.45)

    dominant = max(final_scores, key=final_scores.get)
    sorted_profiles = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "scores": final_scores,
        "sorted": sorted_profiles,
        "dominant": dominant,
        "matched_keywords": matched,
        "avg_sent_len": round(avg_sent_len, 1),
        "unique_ratio": round(unique_ratio * 100, 1),
        "q_count": q_count,
        "passive_hits": passive_hits,
        "word_count": words_total,
        "sentence_count": len(sentences),
    }

def plot_psych_bar(result):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d0d0d")
    names = [p for p, _ in result["sorted"]]
    vals = [v for _, v in result["sorted"]]
    colors = [PSYCH_PROFILES[n]["color"] for n in names]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="none", height=0.55)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color="white", fontsize=10)
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.set_xlabel("Composite Psychology Score", color="white")
    ax.set_title("Writing Psychology Profile", color="#c9f542", fontweight="bold")
    plt.tight_layout()
    return fig

def plot_radar(result):
    labels = list(PSYCH_PROFILES.keys())
    values = [result["scores"][l] for l in labels]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor="#0d0d0d")
    ax.set_facecolor("#111")
    ax.plot(angles, values, color="#c9f542", linewidth=2)
    ax.fill(angles, values, color="#c9f542", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.yaxis.set_tick_params(labelcolor="gray")
    ax.set_title("Keyword Profile Radar", color="#c9f542", pad=20, fontweight="bold")
    plt.tight_layout()
    return fig

def plot_writing_stats(result):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), facecolor="#0d0d0d")
    metrics = [
        ("Avg Sentence\nLength (words)", result["avg_sent_len"], 30, "#42c8f5"),
        ("Vocabulary\nRichness (%)", result["unique_ratio"], 100, "#f542b9"),
        ("Questions\nAsked", result["q_count"], max(result["q_count"] + 1, 5), "#f5a442"),
    ]
    for ax, (label, val, mx, color) in zip(axes, metrics):
        ax.set_facecolor("#111")
        ax.barh([label], [val], color=color, height=0.4)
        ax.set_xlim(0, mx)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_color("#333")
        ax.set_xlabel(label, color="white", fontsize=9)
        ax.set_title(f"{val}", color=color, fontsize=16, fontweight="bold")
    fig.suptitle("Writing Style Metrics", color="white", fontsize=13)
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🧠 NLP Similarity + Writing Psychology Analyzer")
st.caption("**Model:** `sentence-transformers/all-MiniLM-L6-v2` · Free · No preprocessing · No training")

tab1, tab2 = st.tabs(["📊 Text / Word Similarity", "🧬 Writing Psychology Analyzer"])

# ─── TAB 1: Similarity ─────────────────────────────────────────────────────────
with tab1:
    st.header("Text & Word Similarity")
    st.info("Enter multiple words or sentences (one per line). The model computes semantic similarity scores using pretrained embeddings — no preprocessing, no training.")

    default_input = "king\nqueen\nman\nwoman\nroyal\ncastle\nleader\npower"
    user_input = st.text_area("Enter words or sentences (one per line):", value=default_input, height=180)
    items = [line.strip() for line in user_input.strip().split("\n") if line.strip()]

    if len(items) < 2:
        st.warning("Enter at least 2 items.")
    else:
        with st.spinner("Computing embeddings..."):
            embeddings = embed(items)
            sim_matrix = cosine_similarity(embeddings)

        anchor = st.selectbox("Select anchor item for bar chart:", items)
        anchor_idx = items.index(anchor)
        others = [(items[i], sim_matrix[anchor_idx][i]) for i in range(len(items)) if i != anchor_idx]
        others_sorted = sorted(others, key=lambda x: x[1], reverse=True)

        col1, col2 = st.columns(2)

        # Bar Chart
        with col1:
            st.subheader("📊 Bar Chart — Top Similar Items")
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d0d0d")
            labels = [o[0] for o in others_sorted]
            vals = [o[1] for o in others_sorted]
            palette = plt.cm.YlGn(np.linspace(0.4, 0.9, len(labels)))
            ax.barh(labels[::-1], vals[::-1], color=palette[::-1], edgecolor="none")
            ax.set_facecolor("#0d0d0d")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333")
            ax.set_xlabel("Cosine Similarity", color="white")
            ax.set_title(f"Similarity to: '{anchor}'", color="#c9f542", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Critical Thinking Note
            top = others_sorted[0]
            st.markdown(f"""
            <div class="metric-card">
            <b>🧠 Paul's Critical Thinking Standards</b><br><br>
            <b>Clarity:</b> Input was <em>"{anchor}"</em>. Model returned cosine similarity scores showing semantic closeness in embedding space.<br>
            <b>Accuracy:</b> Model used: <code>all-MiniLM-L6-v2</code> — a pretrained sentence transformer. No manual claims added.<br>
            <b>Precision:</b> Top match is <em>"{top[0]}"</em> with score <code>{top[1]:.4f}</code>.<br>
            <b>Logic:</b> "{top[0]}" is closest to "{anchor}" because they likely appear in similar linguistic contexts in the training corpus.<br>
            <b>Significance:</b> Scores above 0.8 indicate strong semantic overlap; below 0.5 suggests conceptual distance.<br>
            <b>Fairness:</b> Limitation — pretrained models reflect biases in training data; similarity depends on corpus used during pretraining.
            </div>
            """, unsafe_allow_html=True)

        # Heatmap
        with col2:
            st.subheader("🌡️ Heatmap — Pairwise Similarity")
            fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor="#0d0d0d")
            sns.heatmap(sim_matrix, xticklabels=items, yticklabels=items,
                        annot=True, fmt=".2f", cmap="YlOrBr",
                        linewidths=0.4, linecolor="#333",
                        ax=ax2, cbar_kws={"shrink": 0.8})
            ax2.set_facecolor("#0d0d0d")
            ax2.tick_params(colors="white", labelsize=8)
            ax2.set_title("Pairwise Cosine Similarity", color="#c9f542", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # PCA Plot
        st.subheader("🗺️ 2D Embedding Plot — PCA")
        n_components = min(2, len(items))
        if len(items) >= 2:
            pca = PCA(n_components=n_components)
            coords = pca.fit_transform(embeddings)
            fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor="#0d0d0d")
            ax3.set_facecolor("#111")
            scatter_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(items)))
            for i, (item, coord) in enumerate(zip(items, coords)):
                ax3.scatter(coord[0], coord[1], color=scatter_colors[i], s=120, zorder=3)
                ax3.annotate(item, (coord[0], coord[1]),
                             textcoords="offset points", xytext=(7, 4),
                             color="white", fontsize=10)
            ax3.spines[:].set_color("#333")
            ax3.tick_params(colors="white")
            ax3.set_xlabel("PC1", color="white")
            ax3.set_ylabel("PC2", color="white")
            ax3.set_title("PCA Projection of Word Embeddings", color="#c9f542", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
            st.caption(f"PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% of total variance. Semantically similar words cluster together.")

        # Relevance note
        st.markdown("""
        <div class="metric-card">
        <b>📌 Relevance:</b> All three graphs — Bar Chart, Heatmap, and PCA plot — directly visualize the cosine similarity scores produced by the model. No extra data was added.
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: Writing Psychology ──────────────────────────────────────────────────
with tab2:
    st.header("🧬 Writing Psychology Analyzer")
    st.markdown("Upload a `.docx` or `.txt` file, or paste text below. The model analyzes your **writing style** to identify your dominant **cognitive/thinking pattern**.")

    input_method = st.radio("Input method:", ["Paste text", "Upload file (.docx / .txt)"], horizontal=True)

    analysis_text = ""
    if input_method == "Paste text":
        analysis_text = st.text_area("Paste your writing here:", height=220, placeholder="Write anything — an essay, notes, answers, reflections...")
    else:
        uploaded = st.file_uploader("Upload file:", type=["docx", "txt"])
        if uploaded:
            if uploaded.name.endswith(".docx"):
                analysis_text = extract_text_from_docx(uploaded)
            else:
                analysis_text = extract_text_from_txt(uploaded)
            with st.expander("📄 Extracted text preview"):
                st.text(analysis_text[:1500] + ("..." if len(analysis_text) > 1500 else ""))

    if st.button("🔍 Analyze Writing Psychology", use_container_width=True):
        if not analysis_text or len(analysis_text.strip()) < 50:
            st.error("Please provide at least 50 characters of text.")
        else:
            with st.spinner("Analyzing writing patterns..."):
                result = analyze_psychology(analysis_text)

            if result:
                dom = result["dominant"]
                dom_color = PSYCH_PROFILES[dom]["color"]
                dom_desc = PSYCH_PROFILES[dom]["description"]

                # Dominant Profile Banner
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-left:5px solid {dom_color};
                border-radius:12px;padding:20px 24px;margin:12px 0;">
                <p style="color:#aaa;font-size:13px;margin:0">DOMINANT THINKING STYLE</p>
                <h2 style="color:{dom_color};margin:4px 0;">{dom}</h2>
                <p style="color:#ddd;margin:0">{dom_desc}</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([1.2, 1])

                with col1:
                    st.subheader("📊 Psychology Score Breakdown")
                    fig_bar = plot_psych_bar(result)
                    st.pyplot(fig_bar)
                    plt.close()

                with col2:
                    st.subheader("🕸️ Keyword Profile Radar")
                    fig_radar = plot_radar(result)
                    st.pyplot(fig_radar)
                    plt.close()

                # Writing Stats
                st.subheader("📝 Writing Style Metrics")
                fig_stats = plot_writing_stats(result)
                st.pyplot(fig_stats)
                plt.close()

                # Matched Keywords
                st.subheader("🔑 Profile-Matched Keywords Found")
                for profile, kws in result["matched_keywords"].items():
                    if kws:
                        color = PSYCH_PROFILES[profile]["color"]
                        kw_html = " ".join([f'<span class="tag-pill">{kw} ×{c}</span>' for kw, c in kws[:8]])
                        st.markdown(f"""
                        <div style="margin:6px 0">
                        <span style="color:{color};font-weight:700">{profile}:</span> {kw_html}
                        </div>
                        """, unsafe_allow_html=True)

                # Paul's Critical Thinking Standards
                st.markdown(f"""
                <div class="metric-card" style="margin-top:18px">
                <b>🧠 Paul's Critical Thinking Standards — Applied to This Analysis</b><br><br>
                <b>Clarity:</b> The text contained {result['word_count']} words across ~{result['sentence_count']} sentences. Analyzed for psychological indicators via keyword density and semantic similarity.<br>
                <b>Accuracy:</b> Scores are derived from keyword frequency (55%) and semantic cosine similarity to profile descriptions (45%) using <code>all-MiniLM-L6-v2</code>.<br>
                <b>Precision:</b> Dominant profile: <em>{dom}</em> — composite score: <code>{result['scores'][dom]:.3f}</code>. Avg sentence length: {result['avg_sent_len']} words. Vocabulary richness: {result['unique_ratio']}%.<br>
                <b>Relevance:</b> All three graphs (bar, radar, metrics) directly represent features extracted from the uploaded text.<br>
                <b>Logic:</b> A "{dom}" profile is inferred because the writing contains vocabulary and semantic patterns strongly associated with {dom_desc.lower()}<br>
                <b>Significance:</b> The top two profiles are <em>{result['sorted'][0][0]}</em> ({result['sorted'][0][1]:.2f}) and <em>{result['sorted'][1][0]}</em> ({result['sorted'][1][1]:.2f}). Blend of these may reflect the writer's true style.<br>
                <b>Fairness:</b> Limitation — this analysis is probabilistic, not clinical. Keyword matching can miss tone, sarcasm, or second-language patterns. Results are indicative, not diagnostic.
                </div>
                """, unsafe_allow_html=True)
