import streamlit as st
import pandas as pd
import numpy as np
import spacy
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI Resume Screener", layout="wide")

# ---------- FUNCTIONS ----------

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = Document(file)
    return " ".join([p.text for p in doc.paragraphs])

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    return " ".join(tokens)

def extract_skills(text, skill_list):
    found = []
    for skill in skill_list:
        if skill in text:
            found.append(skill)
    return list(set(found))

def match_score(resume, jd):
    tfidf = TfidfVectorizer()
    mat = tfidf.fit_transform([resume, jd])
    score = cosine_similarity(mat[0], mat[1])[0][0]
    return round(score*100,2)

# ---------- UI ----------

st.title("AI Resume Screener System")

jd = st.text_area("Paste Job Description", height=200)

uploaded = st.file_uploader(
    "Upload Resumes (PDF/DOCX)", 
    accept_multiple_files=True
)

skill_db = pd.read_csv("skills.csv")["skill"].tolist()

if st.button("Analyze"):
    if jd == "" or not uploaded:
        st.warning("Upload resumes and paste JD")
    else:
        results = []

        for file in uploaded:
            if file.name.endswith(".pdf"):
                text = read_pdf(file)
            else:
                text = read_docx(file)

            clean_resume = clean_text(text)
            clean_jd = clean_text(jd)

            skills = extract_skills(clean_resume, skill_db)
            score = match_score(clean_resume, clean_jd)

            results.append({
                "Candidate": file.name,
                "Match Score (%)": score,
                "Skills Found": ", ".join(skills)
            })

        df = pd.DataFrame(results)
        df = df.sort_values(by="Match Score (%)", ascending=False)

        st.subheader("Ranking")
        st.dataframe(df, use_container_width=True)

        st.subheader("Top Candidate")
        st.success(df.iloc[0]["Candidate"])