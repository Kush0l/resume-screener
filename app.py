import streamlit as st
import google.generativeai as genai
from google import genai

import os
from dotenv import load_dotenv
import pdfplumber
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Replace with your specific model import


load_dotenv() ## load all our environment variables

genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text):
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=input_text,
    )
    return response.text

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def main():
    st.title("Resume Screening App")
    jd = st.text_area("Paste the Job Description")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
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
        Hey, act like a skilled ATS (Applicant Tracking System) with deep understanding of the tech field, software engineering, data science, 
        data analysis, big data engineering, and more. Your task is to evaluate the resume based on the given job description.
        Consider the competitive job market and provide the best assistance for improving the resume.
        Assign the percentage matching based on JD and missing keywords with high accuracy.

        Resume: {resume_text}
        Job Description: {jd}
        """

        st.write("Predicted Category:", predicted_category)
        submit = st.button("Analyze Resume with Job Description")
        
        if submit:
            response = get_gemini_response(input_prompt)
            st.subheader(response)

if __name__ == "__main__":
    main()