import streamlit as st
import pandas as pd
import ollama
import re

# Load data
df = pd.read_csv("trans_look_like_new_final_file.csv")

# Column descriptions
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

# Instantiate Ollama client (change host if you have remote server)
client = ollama.Client(host="http://localhost:11434")  # Or your remote host URL

def analyze_data(user_prompt):
    system_message = (
        "You are an expert retail data analyst. "
        "Created by AMAN JAIN, a well-known Data Scientist at Dunnhumby. "
        "Dataset columns (name: meaning, type, and example values):\n"
        + "\n".join([f"- {k}: {v}" for k, v in column_descriptions.items()])
        + "\nPlease write python code only to analyze the following question on the loaded dataframe `df`. "
        "Do not include any text explanations or print(). "
        "Always assign your answer to a variable named result, e.g., result = ... . "
        "If you filter rows in pandas, always check if the result is empty—if so, set result = 'No matching row'."
    )
    user_message = user_prompt

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    try:
        # Call Ollama chat API using client instance
        response = client.chat(model="llama3.1:8b", messages=messages)
        content = response['message']['content']
    except Exception as e:
        return "", f"Connection error calling Ollama API: {str(e)}"

    # Extract python code block (triple backticks)
    code_match = re.search(r"``````", content)
    if not code_match:
        code_match = re.search(r"``````", content)
    if code_match:
        python_code = code_match.group(1).strip()
    else:
        python_code = content.strip()

    if "result" not in python_code:
        python_code = f"result = {python_code}"

    local_vars = {"df": df.copy()}
    output = "Error: No output generated"
    try:
        exec(python_code, {}, local_vars)
        output = local_vars.get("result", "No result variable found in executed code.")
    except Exception as e:
        output = f"Error executing code: {str(e)}\nGenerated code: {python_code}"
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
