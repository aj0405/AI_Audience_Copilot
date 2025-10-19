import os
import re
import streamlit as st
import pandas as pd
from google import genai

# Load your dataset
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

# Load the API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Environment variable 'GOOGLE_API_KEY' is not set!")

# Initialize the Gemini client (automatic from environment)
client = genai.Client()

# Utility: classify general/casual queries vs data analysis queries
def is_general_query(text):
    general_keywords = [
        "hi", "hello", "how are you", "prime minister", "president", "weather", "temperature",
        "what's up", "who is", "tell me", "chat", "joke", "today", "casual", "news"
    ]
    return any(word in text.lower() for word in general_keywords)

# For general/casual chat with Gemini
def ask_gemini_chat(query, client):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query
    )
    return response.text

# For dataset analysis
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
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    generated_text = response.text
    code_match = re.search(r"``````", generated_text)
    if not code_match:
        code_match = re.search(r"``````", generated_text)
    if code_match:
        python_code = code_match.group(1).strip()
    else:
        python_code = generated_text.strip()
    if "result" not in python_code:
        python_code = f"result = {python_code}"
    local_vars = {"df": df.copy()}
    output = "Error: No output generated"
    try:
        exec(python_code, {}, local_vars)
        output = local_vars.get("result", "No result variable found in executed code.")
    except Exception as e:
        output = f"Error executing code: {str(e)}\nGenerated code: {python_code}"
    return python_code, output

# Streamlit UI
st.set_page_config(page_title="Personal AI Data Copilot", layout="wide")
st.title("Personal AI Data Copilot")
st.markdown(
    "Ask questions about your transactional dataset in natural language. "
    "The AI generates and runs pandas code on your data to answer your queries. "
    "You can also ask general questions, like the weather or latest news!"
)

query = st.text_area("Enter your analysis question", height=120)

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        if is_general_query(query):
            # General/casual query handled by Gemini chat
            ai_response = ask_gemini_chat(query, client)
            st.subheader("AI Response")
            st.write(ai_response)
        else:
            # Data analysis using pandas/Gemini
            python_code, ai_output = analyze_data(query)
            st.subheader("Generated Python Code")
            st.code(python_code, language="python")
            st.subheader("Analysis Output")
            # Scrollable pane for large outputs
            if isinstance(ai_output, pd.DataFrame):
                st.dataframe(ai_output, use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div style="overflow-x:auto; overflow-y:auto; max-height:400px; max-width:100%;">
                        <pre>{ai_output}</pre>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
