# TalentLens — Resume Screener

TalentLens is a polished Streamlit workspace for first-pass resume screening. Paste a job description, upload PDF or DOCX resumes, and review a ranked shortlist with transparent relevance scores and detected skills.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app uses the bundled spaCy English model when available and falls back to a lightweight tokenizer so the interface remains usable when the model is unavailable.

## Screening logic

- Skills use whole-word regular-expression matching, so `java` does not match `javascript` and `git` does not match `digital`.
- TF-IDF is fitted once on all cleaned resumes plus the job description. Each resume is then scored against the same job-description vector, making scores comparable across the candidate pool.
- Scores are a screening aid, not a hiring decision. Human review should remain part of the process.
