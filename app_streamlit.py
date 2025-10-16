import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# Load your dataset
df = pd.read_csv("trans_look_like_new_final_file.csv")

# Column descriptions dictionary
column_descriptions = {
    "PRODUCT_GROUP_ID": "A unique number that groups similar products together.",
    "TXN_BASKET_KEY": "A code that represents a customer's shopping basket or visit.",
    "HOUSEHOLD_KEY": "A unique number that identifies each customer household.",
    "PRODUCT_KEY": "A unique number given to a specific product.",
    "SALES_CHANNEL_ID": "Shows how the sale was made: 'STO' means in-store, 'ONL' means online.",
    "STORE_LOCATION_CODE": "Code that tells which store the sale happened at.",
    "LOYALTY_FLAG": "Shows if a loyalty card was used in the purchase: 'Y' means yes, 'N' means no.",
    "QUANTITY_PURCHASED": "How many items were bought.",
    "NET_EXPENDITURE": "The amount spent after discounts or coupons.",
    "TRANSACTION_DATETIME": "The exact date and time when the purchase was made.",
    "FINANCIAL_YEAR_KEY": "The year according to the retailer's calendar.",
    "FINANCIAL_WEEK_KEY": "The week number according to the retailer's calendar.",
    "product": "Description of the product, usually including name and size.",
    "BRAND": "Name of the brand of the product.",
    "category": "The main category the product belongs to, like Grocery or Beers Wines and Spirits."
}

# Load model via pipeline
@st.cache_resource(show_spinner=False)
def load_model_pipeline():
    return pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

pipe = load_model_pipeline()

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

    response = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, 
                    pad_token_id=pipe.tokenizer.eos_token_id, eos_token_id=pipe.tokenizer.eos_token_id)
    generated_text = response[0]['generated_text']

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

# Streamlit UI
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

