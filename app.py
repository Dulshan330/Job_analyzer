import streamlit as st
import ollama
import json
from pydantic import BaseModel, Field
from typing import List
import pypdf
import docx

# --- Configuration & Constants ---
MODEL_NAME = "llama3.1:8b"

# --- Data Models (Pydantic) ---
# This ensures the LLM output adheres to a strict schema for production stability
class AnalysisResult(BaseModel):
    matching_score: int = Field(description="A score from 0 to 100 representing the fit")
    existing_skills: List[str] = Field(description="List of skills found in resume matching the JD")
    missing_skills: List[str] = Field(description="List of critical skills present in JD but missing in Resume")
    recommended_improvements: List[str] = Field(description="Actionable advice to improve the resume")
    suitability_status: str = Field(description="One of: 'Highly Recommended', 'Qualified', 'Potential Match', 'Not Qualified'")

# --- Helper Functions ---
def extract_text_from_pdf(file):
    pdf = pypdf.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# --- Backend Logic ---
def get_ollama_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Sends the resume and JD to the local Ollama instance for analysis.
    Returns a dictionary matching the AnalysisResult schema.
    """

    # specialized system prompt for the role
    system_prompt = """
    You are a strict, evidence-based AI Career Coach and ATS (Applicant Tracking System) specialist. 
    Your goal is to compare a candidate's resume against a job description (JD) based ONLY on the provided text.

    CRITICAL RULES:
    1. **NO HALLUCINATIONS:** Do not assume the candidate has a skill unless it is explicitly stated in the Resume text. Even if a candidate is a "Senior Developer," do not assume they know "Docker" unless the word "Docker" appears in the resume.
    2. **Existing Skills:** This list must ONLY contain technical skills and keywords that appear in BOTH the JD and the Resume.
    3. **Missing Skills:** This list must contain critical skills found in the JD that are completely ABSENT from the Resume.
    4. **Synonyms:** You may recognize standard synonyms (e.g., "React" = "React.js", "AWS" = "Amazon Web Services"), but if the concept is missing, mark it as missing.
    5. **Output Format:** You must output ONLY valid JSON.
    """

    user_prompt = f"""
    Please perform a deep gap analysis on the following data:
    
    **Job Description:**
    {jd_text}

    **Resume:**
    {resume_text}

    Return the analysis in the following JSON format:
    {{
        "matching_score": <integer>,
        "existing_skills": [<list of strings>],
        "missing_skills": [<list of strings>],
        "recommended_improvements": [<list of strings>],
        "suitability_status": "<One of: Highly Recommended, Qualified, Potential Match, Not Qualified>"
    }}
    """

    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ], format='json') # Enforce JSON mode in Llama 3.1

        content = response['message']['content']
        
        # Validation step using Pydantic
        data = json.loads(content)
        validated_data = AnalysisResult(**data)
        return validated_data.model_dump()

    except ollama.ResponseError as e:
        st.error(f"Ollama API Error: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Failed to parse LLM response. The model did not return valid JSON.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Frontend UI (Streamlit) ---
def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

    # Header
    st.title("📄 AI Resume & Job Matcher")
    st.markdown("---")

    # Layout: Two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Paste Job Description")
        jd_input = st.text_area("Copy/Paste the JD here", height=300, placeholder="e.g. Senior Python Engineer...")

    with col2:
        st.subheader("2. Upload Resume")
        uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
        resume_text = ""
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = extract_text_from_docx(uploaded_file)
            except Exception as e:
                st.error(f"Error extracting text: {e}")

    # Action Button
    analyze_btn = st.button("Analyze Resume", type="primary", use_container_width=True)

    if analyze_btn:
        if not jd_input or not resume_text:
            st.warning("Please provide both the Job Description and the Resume.")
        else:
            with st.spinner("Consulting the AI agent..."):
                result = get_ollama_analysis(resume_text, jd_input)

            if result:
                # --- Display Results ---
                st.divider()
                st.header("Analysis Results")

                # Top Metrics Row
                m_col1, m_col2 = st.columns([1, 3])
                
                with m_col1:
                    score = result['matching_score']
                    # Color coding the score
                    delta_color = "normal"
                    if score >= 80: delta_color = "normal" 
                    elif score < 50: delta_color = "inverse"
                    
                    st.metric(label="Matching Score", value=f"{score}%", delta=result['suitability_status'], delta_color=delta_color)
                    
                    # Status Badge
                    if result['suitability_status'] == 'Highly Recommended':
                        st.success(result['suitability_status'])
                    elif result['suitability_status'] == 'Not Qualified':
                        st.error(result['suitability_status'])
                    else:
                        st.info(result['suitability_status'])

                with m_col2:
                    st.subheader("💡 Recommended Improvements")
                    for tip in result['recommended_improvements']:
                        st.markdown(f"- {tip}")

                # Skills Comparison Columns
                st.divider()
                s_col1, s_col2 = st.columns(2)

                with s_col1:
                    st.subheader("✅ Existing Skills")
                    if result['existing_skills']:
                        # Display as pills or tags
                        st.markdown(" ".join([f"`{skill}`" for skill in result['existing_skills']]))
                    else:
                        st.write("No matching skills found.")

                with s_col2:
                    st.subheader("⚠️ Missing / Gap Skills")
                    if result['missing_skills']:
                        st.markdown(" ".join([f"**{skill},**" for skill in result['missing_skills']])) # Bold red-ish style
                        st.warning("Adding these skills to your resume contextually will increase your skill matching score.")
                    else:
                        st.success("No major skill gaps detected!")

if __name__ == "__main__":
    main()