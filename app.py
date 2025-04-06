import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pdfplumber
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from streamlit.components.v1 import html

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

# Resume cleaning function
def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

# PDF extractor
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

# Gemini API call
def get_gemini_response(input_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(input_text)
    return response.text

# Streamlit app
def main():
    st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="wide")
    st.title("📄 Resume Screening & Analysis Tool")

    with st.sidebar:
        st.image("https://source.unsplash.com/400x200/?resume,job")
        st.header("Upload & Analyze Your Resume")
        uploaded_file = st.file_uploader("Upload Resume (PDF/TXT)", type=["pdf", "txt"])
        jd = st.text_area("Paste the Job Description")
        analyze_button = st.button("🔍 Analyze Resume")

    if uploaded_file and jd and analyze_button:
        with st.spinner("Processing your resume..."):
            resume_text = ""
            if uploaded_file.type == "text/plain":
                resume_text = uploaded_file.read().decode('utf-8')
            elif uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf_vectorizer.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            category_mapping = {
                15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer", 24: "Web Designing", 
                12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer", 18: "Operations Manager", 6: "Data Science", 
                22: "Sales", 16: "Mechanical Engineer", 1: "Arts", 7: "Database", 11: "Electrical Engineering", 
                14: "Health and Fitness", 19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing", 
                17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
            }

            predicted_category = category_mapping.get(prediction_id, "Unknown")

            input_prompt = f"""
You are an advanced Applicant Tracking System (ATS) assistant for technical roles (e.g., Software Engineer, Data Scientist).

## Task:
Analyze the given resume and job description. Return a fully styled HTML report with:

1. 👤 **Candidate Information** (Name, Email, Phone — if available)
2. 🧠 **Predicted Job Category** (from: {predicted_category})
3. 📊 **Match Percentage** (based on keyword and skill overlap)
4. ❌ **Missing Keywords** (important skills not found in resume)
5. ✅ **Suggestions to Improve Resume** (bullet points)
6. 🛠️ **Skills Extracted from Resume**
7. 📁 **Projects Mentioned**
8. 💼 **Past Work Experience**

## Output format:
- ONLY return **pure HTML**.
- DO NOT use triple backticks or markdown like ```html.
- Include inline CSS (no external stylesheets).
- Must be clean, professional, and modern in **light mode**.
- Use white background, black/dark text, blue/green highlights.

## Validation Rule:
If the resume or job description does not contain enough relevant information (e.g., too short, irrelevant content, missing technical context), respond ONLY with:
Invalid job description or Resume

## Sample layout style:
<div style="font-family:'Segoe UI',sans-serif; background-color:#fff; padding:24px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.1); color:#202124; line-height:1.6;">

  <h2 style="color:#1a73e8;">📄 Resume Analysis Report</h2>

  <div style="margin-bottom:20px;">
    <h3>👤 Candidate Info</h3>
    <p><strong>Name:</strong> Jane Doe<br>
       <strong>Email:</strong> jane@example.com<br>
       <strong>Phone:</strong> +1-234-567-8901</p>
  </div>

  <div style="margin-bottom:20px;">
    <p><strong>Predicted Job Category:</strong> <span style="color:#0f9d58;">{predicted_category}</span></p>
    <p><strong>Match Percentage:</strong> <span style="background-color:#e8f0fe; padding:6px 10px; border-radius:5px;">88%</span></p>
  </div>

  <div style="margin-bottom:20px;">
    <h3>🛠️ Skills</h3>
    <ul><li>Python</li><li>Flask</li><li>MongoDB</li><li>Machine Learning</li></ul>
  </div>

  <div style="margin-bottom:20px;">
    <h3>📁 Projects</h3>
    <ul><li><strong>Resume Analyzer:</strong> Built with Streamlit & Gemini, analyzed resumes against JDs.</li><li><strong>Stock Price Predictor:</strong> LSTM-based forecasting app using TensorFlow.</li></ul>
  </div>

  <div style="margin-bottom:20px;">
    <h3>💼 Past Experience</h3>
    <ul><li><strong>Software Intern at ABC Corp</strong> (Jun 2023 - Dec 2023)</li><li><strong>Freelance Web Developer</strong> (2022 - 2023)</li></ul>
  </div>

  <div style="margin-bottom:20px;">
    <h3>❌ Missing Keywords</h3>
    <ul><li>PostgreSQL</li><li>Docker</li></ul>
  </div>

  <div style="margin-bottom:20px;">
    <h3>✅ Suggestions</h3>
    <ul><li>Add PostgreSQL experience in backend projects.</li><li>Include deployment stack/tools like Docker or Kubernetes.</li></ul>
  </div>

</div>

## Resume:
{resume_text}

## Job Description:
{jd}

## IMPORTANT:
- DO NOT wrap the response in triple backticks or markdown code blocks.
- ONLY return clean HTML as per the format above.
- No explanations or text outside the HTML.
- If any section is not present in the resume like experience, projects, etc. Give <p>Not Found</p> with styling.
"""


            # Get Gemini response
            response_html = get_gemini_response(input_prompt)

            # Show predicted category
            st.success("✅ Analysis Complete!")
            st.subheader("📌 Predicted Job Category of Resume")
            st.info(f"**{predicted_category}**")

            # Show Gemini analysis
            st.subheader("📊 AI Resume Analysis")
            html(response_html, height=650, scrolling=True)

if __name__ == "__main__":
    main()
