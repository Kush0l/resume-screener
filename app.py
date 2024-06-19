import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pdfplumber
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Replace with your specific model import


load_dotenv() ## load all our environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_repsonse(input):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(input)
    return response.text

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    # Your cleaning logic here (similar to previous examples)
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
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def main():
    st.title("Resume Screening App")
    jd=st.text_area("Paste the Job Description")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        resume_text = ""

        # Handle text file
        if uploaded_file.type == "text/plain":
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        
        # Handle PDF file
        elif uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        
        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf_vectorizer.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

                # Mapping of category IDs to names
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        predicted_category = category_mapping.get(prediction_id, "Unknown")


        input_prompt=f"""
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering,data science ,data analyst
and big data engineer and many more. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving thr resumes. Assign the percentage Matching based 
on Jd and the missing keywords with high accuracy
resume:{resume_text}
description:{jd}

I want the response in one single string having the below structure :- 

`***------- `Resume Analysis with Job Description` -------***`


## CV Match : "%" \n
### Category of Resume :--> `{predicted_category}`
\n
## **`MissingKeywords:`** \n  
1. keyword 1, \n
2. keyword 2, \n
3. keyword 3, \n
4. .., \n
5. .., \n 
\n

## **`Profile Summary:`** \n
 ### **`Name` : NameOfApplicant** \n
\n
 ### **`Skills :`** \n
    1. Skill 1 Of Applicant  \n
    2. Skill 2 Of Applicant  \n
    3. Skill 3 Of Applicant  \n
    4. Skill 4 Of Applicant  \n
    5. Skill 5 Of Applicant  \n

 ### **`Projects :`** \n
    1. Project 1 Of Applicant  \n
    2. Project 2 Of Applicant  \n
    3. Project 3 Of Applicant  \n
    4. Project 4 Of Applicant  \n

 ### **`Work Experince :`** \n
    1. Work experince 1 Of Applicant  \n
    2. Work experince 2 Of Applicant  \n
    3. Work experince 3 Of Applicant  \n
    4. Work experince 4 Of Applicant  \n
\n
## **`How Resume can be Improved to match the Job Description:`**\n
 //Explain how applicante can improve to match the job description
"""

        st.write("Predicted Category:", predicted_category)
        submit = st.button("Analyze Resume with Job Description")
        if submit:
            if uploaded_file is not None:
                response=get_gemini_repsonse(input_prompt)
                st.subheader(response)




if __name__ == "__main__":
    main()



























# import streamlit as st
# import pickle
# import re
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier  # Replace with your specific model import

# # Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# # Load models
# clf = pickle.load(open('clf.pkl','rb'))
# tfidf_vectorizer = pickle.load(open('tfidf.pkl','rb'))

# # Function to clean resume text
# def clean_resume(resume_text):
#     # Clean resume text using regular expressions
#     clean_text = re.sub('http\S+\s*', ' ', resume_text)
#     clean_text = re.sub('RT|cc', ' ', clean_text)
#     clean_text = re.sub('#\S+', '', clean_text)
#     clean_text = re.sub('@\S+', '  ', clean_text)
#     clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
#     clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
#     clean_text = re.sub('\s+', ' ', clean_text)
#     return clean_text

# # Main function for Streamlit app
# def main():
#     st.title("Resume Screening App")
#     uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

#     if uploaded_file is not None:
#         try:
#             resume_bytes = uploaded_file.read()
#             resume_text = resume_bytes.decode('utf-8')
#         except UnicodeDecodeError:
#             resume_text = resume_bytes.decode('latin-1')

#         cleaned_resume = clean_resume(resume_text)  # Corrected: clean_resume instead of cleaned_text
#         input_features = tfidf_vectorizer.transform([cleaned_resume])
#         prediction_id = clf.predict(input_features)[0]

#         # Mapping of category IDs to names
#         category_mapping = {
#             15: "Java Developer",
#             23: "Testing",
#             8: "DevOps Engineer",
#             20: "Python Developer",
#             24: "Web Designing",
#             12: "HR",
#             13: "Hadoop",
#             3: "Blockchain",
#             10: "ETL Developer",
#             18: "Operations Manager",
#             6: "Data Science",
#             22: "Sales",
#             16: "Mechanical Engineer",
#             1: "Arts",
#             7: "Database",
#             11: "Electrical Engineering",
#             14: "Health and fitness",
#             19: "PMO",
#             4: "Business Analyst",
#             9: "DotNet Developer",
#             2: "Automation Testing",
#             17: "Network Security Engineer",
#             21: "SAP Developer",
#             5: "Civil Engineer",
#             0: "Advocate",
#         }


#         predicted_category = category_mapping.get(prediction_id, "Unknown")

#         st.write("Predicted Category:", predicted_category)

# # Entry point of the script
# if __name__ == "__main__":
#     main()
