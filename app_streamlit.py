import streamlit as st
import pandas as pd
import re
import os
from google import genai

# Load your dataset
df = pd.read_csv("trans_look_like_new_final_file.csv")

# Column descriptions dictionary (same as before)
column_descriptions = {
    # ... (keep your existing descriptions)
}

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini client
client = genai.Client()

def analyze_data(user_prompt):
    system_message = (
        "You are an expert retail data analyst. "
        "Created by AMAN JAIN, a well-known Data Scientist at Dunnhumby. "
        "Dataset columns (name: meaning):\n"
        + "\n".join([f"- {k}: {v}" for k, v in column_descriptions.items()])
        + "\nPlease write python code only to analyze the following question on the loaded dataframe `df`. "
        "Do not include any text explanations or print(). "
        "Always assign your answer to a variable named result, e.g., result = ... . "
        "If you filter rows in pandas, always check if the result is empty—if so, set result = 'No matching row'."
    )

    prompt = system_message + "\nUser question: " + user_prompt + "\nPython code:"

    # Use a Gemini model like "gemini-2.5-flash"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        temperature=0.7,
        max_output_tokens=256
    )

    generated_text = response.text

    # Extract Python code from model output
    code_match = re.search(r"``````", generated_text, re.DOTALL)
    if code_match:
        python_code = code_match.group(1).strip()
    else:
        python_code = generated_text.split("Python code:")[-1].strip()

    if "result" not in python_code:
        python_code = f"result = {python_code}"

    local_vars = {"df": df.copy()}
    try:
        exec(python_code, {}, local_vars)
        output = local_vars.get("result", "No result found")
    except Exception as e:
        output = f"Error executing generated code: {str(e)}"

    return python_code, str(output)

# Streamlit UI (same as before)
st.set_page_config(page_title="Personal AI Data Copilot", layout="wide")
st.title("Personal AI Data Copilot")
st.markdown(
    "Ask questions about your transactional dataset in natural language. "
    "The AI generates and runs pandas code on your data to answer your queries, and shows the backend code."
)

query = st.text_area("Enter your analysis question", height=120)

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        python_code, ai_output = analyze_data(query)
    st.subheader("Generated Python Code")
    st.code(python_code, language="python")
    st.subheader("AI Analysis Result")
    st.text(ai_output)
