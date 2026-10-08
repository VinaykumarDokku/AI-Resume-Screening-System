
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")
st.caption("Developed by Vinay Kumar Dokku")

st.write(
    "Upload your resume and compare it with "
    "a job description using NLP."
)

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=200
)

def extract_text(file):
    reader = PdfReader(file)
    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

def calculate_similarity(resume, job):
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )
    vectors = vectorizer.fit_transform([resume, job])
    similarity = cosine_similarity(
        vectors[0:1], vectors[1:2]
    )[0][0]
    return round(float(similarity) * 100, 2)

if st.button("Analyze Resume", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a PDF resume.")
    elif not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        try:
            resume_text = extract_text(uploaded_file)

            if not resume_text.strip():
                st.error("No readable text found in the PDF.")
            else:
                score = calculate_similarity(
                    resume_text, job_description
                )

                st.subheader("Analysis Results")
                st.metric(
                    "Text Similarity Score",
                    f"{score}%"
                )
                st.progress(score / 100)

                st.info(
                    "This measures text similarity, "
                    "not candidate suitability. "
                    "Human review is required."
                )
        except Exception as error:
            st.error(f"Analysis failed: {error}")
