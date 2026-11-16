from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import spacy
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="TalentLens | Resume Screener",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#172033; --muted:#68738a; --line:#e5e9f1; --soft:#f6f8fc; --blue:#4461ee; --blue-soft:#eef1ff; --mint:#dff7ed; --mint-ink:#147a59; }
    .stApp { background:#fbfcfe; color:var(--ink); }
    .block-container { max-width:1260px; padding:2.5rem 3.2rem 4rem; }
    [data-testid="stSidebar"] { background:#f7f8fc; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding:2rem 1.35rem; }
    h1,h2,h3,h4,p,label,button,input,textarea { font-family:'DM Sans',sans-serif; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.03em; color:var(--ink); }
    h1 { font-size:clamp(2rem,3.4vw,3.6rem); line-height:1.02; margin:.35rem 0 1rem; }
    h2 { font-size:1.45rem; }
    .brand { display:flex; align-items:center; gap:.72rem; margin-bottom:2.6rem; }
    .brand-mark { width:38px; height:38px; border-radius:12px; background:var(--ink); color:white; display:grid; place-items:center; font:700 1.2rem 'Space Grotesk'; }
    .brand-name { font:700 1.1rem 'Space Grotesk'; color:var(--ink); }
    .brand-sub { font-size:.72rem; color:var(--muted); margin-top:2px; }
    .eyebrow { display:inline-flex; align-items:center; gap:.45rem; color:var(--blue); background:var(--blue-soft); border-radius:999px; padding:.38rem .72rem; font-size:.75rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .hero-copy { max-width:720px; }
    .hero-copy p { color:var(--muted); font-size:1.05rem; line-height:1.65; max-width:690px; }
    .surface { background:white; border:1px solid var(--line); border-radius:20px; padding:1.35rem 1.5rem; box-shadow:0 10px 30px rgba(23,32,51,.035); }
    .surface-title { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.65rem; }
    .surface-title h3 { margin:0; font-size:1.1rem; }
    .helper { color:var(--muted); font-size:.86rem; line-height:1.5; }
    .step { display:flex; gap:.75rem; margin:1.15rem 0; align-items:flex-start; }
    .step-number { flex:0 0 26px; height:26px; border-radius:50%; display:grid; place-items:center; color:var(--blue); background:var(--blue-soft); font-size:.75rem; font-weight:700; }
    .step strong { display:block; font-size:.86rem; color:var(--ink); }
    .step span { display:block; color:var(--muted); font-size:.78rem; line-height:1.4; margin-top:.18rem; }
    .metric-card { background:white; border:1px solid var(--line); border-radius:17px; padding:1.1rem 1.2rem; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
    .metric-value { color:var(--ink); font:700 1.7rem 'Space Grotesk'; margin-top:.42rem; }
    .candidate-card { background:white; border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem; margin:.7rem 0; }
    .candidate-top { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
    .candidate-name { color:var(--ink); font-weight:700; font-size:1rem; overflow-wrap:anywhere; }
    .score-pill { border-radius:999px; padding:.32rem .62rem; background:var(--mint); color:var(--mint-ink); font-size:.78rem; font-weight:700; white-space:nowrap; }
    .skill-pill { display:inline-block; background:var(--soft); border:1px solid var(--line); color:#4e5a72; padding:.28rem .48rem; border-radius:7px; font-size:.72rem; margin:.3rem .25rem 0 0; }
    .empty-state { border:1px dashed #cbd3e2; border-radius:20px; padding:3rem 1.5rem; text-align:center; background:linear-gradient(135deg,#fff,#f7f9ff); }
    .empty-icon { font-size:2.2rem; margin-bottom:.6rem; }
    .empty-state h3 { margin:.3rem 0; }
    .empty-state p { color:var(--muted); margin:0 auto; max-width:540px; line-height:1.55; }
    .stButton > button[kind="primary"] { background:var(--ink); border:0; border-radius:11px; min-height:3rem; font-weight:700; }
    .stButton > button[kind="primary"]:hover { background:var(--blue); }
    .stTextArea textarea, .stFileUploader section { border-radius:12px; border-color:var(--line); }
    .stProgress > div > div > div { background:var(--blue); }
    .fine-print { color:var(--muted); font-size:.74rem; text-align:center; margin-top:1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_nlp():
    """Use the bundled model when available, otherwise keep the app usable."""
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return spacy.blank("en")


def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(file) -> str:
    doc = Document(file)
    return " ".join(paragraph.text for paragraph in doc.paragraphs)


def clean_text(text: str) -> str:
    doc = load_nlp()(text.lower())
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num:
            continue
        value = token.lemma_.strip() or token.text.strip()
        if value and value != "-pron-":
            tokens.append(value)
    return " ".join(tokens)


def extract_skills(text: str, skill_list: list[str]) -> list[str]:
    """Match skill phrases as whole words, avoiding java/javascript and git/digital false positives."""
    normalized = text.lower()
    found = []
    for skill in skill_list:
        pattern = rf"\b{re.escape(skill.strip().lower())}\b"
        if re.search(pattern, normalized):
            found.append(skill)
    return found


def score_resumes(resumes: list[str], job_description: str) -> list[float]:
    """Fit TF-IDF once on every resume plus the JD, then score each resume against the JD."""
    documents = resumes + [job_description]
    if not any(document.strip() for document in documents):
        return [0.0] * len(resumes)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(documents)
    jd_vector = matrix[-1]
    return [round(float(cosine_similarity(matrix[index], jd_vector)[0][0] * 100), 2) for index in range(len(resumes))]


def read_resume(file) -> str:
    if file.name.lower().endswith(".pdf"):
        return read_pdf(file)
    return read_docx(file)


def render_candidate_card(row: dict, rank: int):
    skills = row["Skills Found"]
    skill_html = "".join(f'<span class="skill-pill">{skill}</span>' for skill in skills) or '<span class="helper">No tracked skills detected</span>'
    st.markdown(
        f"""
        <div class="candidate-card">
            <div class="candidate-top">
                <div class="candidate-name">{rank:02d} &nbsp; {row['Candidate']}</div>
                <div class="score-pill">{row['Match Score (%)']:.1f}% match</div>
            </div>
            <div style="margin-top:.65rem">{skill_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(int(round(row["Match Score (%)"])), 100), text="Relevance score")


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">TalentLens</div><div class="brand-sub">Resume intelligence workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**How it works**")
    st.markdown('<div class="step"><div class="step-number">1</div><div><strong>Add the role</strong><span>Paste the job description and its must-have signals.</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="step"><div class="step-number">2</div><div><strong>Upload candidates</strong><span>Bring PDF or DOCX resumes into one workspace.</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="step"><div class="step-number">3</div><div><strong>Shortlist with confidence</strong><span>Review transparent relevance scores and skills.</span></div></div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Scoring uses corpus-wide TF-IDF similarity. Skill tags use whole-word matching.")


st.markdown('<div class="eyebrow">✦ Hiring signal, without the noise</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy"><h1>Find the resumes that deserve a closer look.</h1><p>Turn a role brief and a stack of resumes into a clear, explainable shortlist. Built for fast first-pass screening—not black-box decisions.</p></div>', unsafe_allow_html=True)

input_col, context_col = st.columns([1.25, .75], gap="large")
with input_col:
    st.markdown('<div class="surface"><div class="surface-title"><h3>Role brief</h3><span class="helper">Required</span></div>', unsafe_allow_html=True)
    jd = st.text_area("Job description", height=230, placeholder="Paste the responsibilities, requirements, and skills for this role…", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
with context_col:
    st.markdown('<div class="surface"><div class="surface-title"><h3>Candidate pool</h3><span class="helper">PDF or DOCX</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload resumes", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        st.success(f"{len(uploaded)} resume{'s' if len(uploaded) != 1 else ''} ready to screen")
    else:
        st.markdown('<p class="helper">Drop one or more resumes here. You can screen a whole shortlist in one pass.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
analyze = st.button("Run screening  →", type="primary", use_container_width=True)

if analyze:
    if not jd.strip() or not uploaded:
        st.warning("Add a job description and at least one resume to run the screening.")
    else:
        skill_db = pd.read_csv(Path(__file__).with_name("skills.csv"))["skill"].dropna().astype(str).tolist()
        with st.spinner("Reading resumes and calculating relevance…"):
            resume_texts = [read_resume(file) for file in uploaded]
            clean_resumes = [clean_text(text) for text in resume_texts]
            clean_jd = clean_text(jd)
            scores = score_resumes(clean_resumes, clean_jd)
            results = []
            for file, text, score in zip(uploaded, clean_resumes, scores):
                results.append({
                    "Candidate": file.name,
                    "Match Score (%)": score,
                    "Skills Found": extract_skills(text, skill_db),
                })
            results.sort(key=lambda item: item["Match Score (%)"], reverse=True)
            st.session_state["screening_results"] = results

results = st.session_state.get("screening_results")
if results:
    st.markdown("<br>", unsafe_allow_html=True)
    best = results[0]
    total_skills = len(set(skill for result in results for skill in result["Skills Found"]))
    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Candidates screened</div><div class="metric-value">{len(results)}</div></div>', unsafe_allow_html=True)
    with metric_b:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Top match</div><div class="metric-value">{best["Match Score (%)"]:.1f}%</div></div>', unsafe_allow_html=True)
    with metric_c:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Skills surfaced</div><div class="metric-value">{total_skills}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    results_col, detail_col = st.columns([1.2, .8], gap="large")
    with results_col:
        st.markdown("### Shortlist ranking")
        st.caption("Higher scores indicate stronger language overlap with the role brief across this candidate pool.")
        for index, result in enumerate(results, start=1):
            render_candidate_card(result, index)
    with detail_col:
        st.markdown("### Recruiter notes")
        st.markdown(
            f'<div class="surface"><div class="surface-title"><h3>Recommended first look</h3><span class="score-pill">{best["Match Score (%)"]:.1f}%</span></div><p class="helper">Start with <strong>{best["Candidate"]}</strong>. The score is calculated against the full uploaded pool, so it is comparable across this screening run.</p><hr style="border:0;border-top:1px solid #e5e9f1"><p class="helper"><strong>Detected signals</strong></p><p class="helper">{", ".join(best["Skills Found"]) if best["Skills Found"] else "No tracked skills detected in the resume text."}</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Screening integrity")
        st.info("Skill tags use whole-word matching, so ‘java’ will not match ‘javascript’ and ‘git’ will not match ‘digital’. TF-IDF is fitted once across every resume plus the job description.")
else:
    st.markdown('<div class="empty-state"><div class="empty-icon">◌</div><h3>Your shortlist will appear here</h3><p>Paste the role brief, upload the candidate resumes, and run a screening pass to see ranked, explainable results.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="fine-print">TalentLens is a screening aid. Keep human review in the loop for every hiring decision.</div>', unsafe_allow_html=True)
