import streamlit as st
import pandas as pd
import json
import re
import urllib.request

def generate_python_code(prompt: str, model_name: str, api_key: str) -> str:
    """Calls Gemini REST API directly using standard urllib."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                result_data = json.loads(response.read().decode('utf-8'))
                content = result_data["candidates"][0]["content"]["parts"][0]["text"]
                # Extract python code block
                match = re.search(r'```python(.*?)```', content, re.DOTALL)
                if match:
                    return match.group(1).strip()
                return content.strip()
            else:
                raise Exception(f"API Error: {response.status}")
    except Exception as e:
        raise Exception(f"Request failed: {str(e)}")

st.set_page_config(layout="wide", page_title="Universal Data AI")
st.markdown(
    """
    <div style='background-color: #f4f6f9; padding: 8px; border-radius: 6px; text-align: center; margin-bottom: 10px; border: 1px solid #e0e4e8;'>
        <span style='font-size: 0.9em;'>✨ Inspired by <a href='https://github.com/nipunbatra/vayuchat-webllm' target='_blank' style='text-decoration: none; font-weight: bold; color: #0366d6;'>VayuChat-WebLLM</a></span>
    </div>
    """,
    unsafe_allow_html=True
)
st.title("Universal Data AI (WebAssembly)")
st.markdown("This app runs **entirely in your browser**! Upload a CSV file and ask questions.")

st.sidebar.title("API Configuration")
user_api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password", help="Get a free key from Google AI Studio")

st.sidebar.title("Model Configuration")
selected_model = st.sidebar.selectbox(
    "Select Gemini Model:",
    [
        "gemini-3.5-flash-lite", 
        "gemma-4-26b-a4b-it",
        "gemini-3.5-flash", 
        "gemini-2.5-flash", 
        "gemini-2.5-pro", 
        "gemini-flash-latest",
        "gemini-pro-latest"
    ],
    index=0
)

st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {uploaded_file.name} successfully!")
    
    with st.expander("Preview Dataset"):
        st.dataframe(df.head())

    # Context File Management
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Context")
    saved_context = st.sidebar.text_area(
        "Edit persistent context instructions for the AI:", 
        value="Return the final answer strictly as a pandas dataframe (e.g. result = summary_df). Streamlit will render it automatically.",
        height=150
    )

    query = st.text_input("Ask a question about your data:")

    if st.button("Analyze") and query:
        if not user_api_key:
            st.error("Please enter a Gemini API Key in the sidebar to run analysis.")
        else:
            with st.spinner("Thinking (Gemini is writing Python code locally)..."):
                # We bypass PandasAI entirely and use a lightweight custom code generator
                system_prompt = f"""
You are a Python data analyst. I have loaded a pandas dataframe named `df`.
Write Python code to answer this question: {query}

User Context:
{saved_context}

CRITICAL RULES:
- If I ask about a specific site (BMC Construction Data), you MUST use a combination of substring matching AND fuzzy matching to find it. Example logic:
  1. try `df['site_name'].str.contains(query, case=False, na=False)`
  2. if empty, use `import difflib; difflib.get_close_matches(query.lower(), [str(x).lower() for x in df['site_name'].dropna().unique()], cutoff=0.3)` and map it back.
- If you need to display metadata alongside statistics, DO NOT try to merge them into `result`. Instead, `import streamlit as st` and call `st.write('### Metadata')` and `st.dataframe(metadata_df)` directly in your code! Then assign the summary statistics to `result`.
- YOU MUST DECLARE A `result` VARIABLE AT THE END OF YOUR SCRIPT containing your final output (e.g., `result = filtered_df`). 
- If you create a plot, YOU MUST USE STREAMLIT NATIVE CHARTS (e.g., `st.bar_chart(df)`, `st.line_chart(df)`) OR explicitly create a matplotlib figure via `fig, ax = plt.subplots()` and display it using `st.pyplot(fig)`. NEVER use `plt.show()`! Assign a success message to `result` if plotting.
- If you use `.agg()`, NEVER use the string '50%'. Use 'median' instead!
- If you need to reset the index of a dataframe, you MUST use `.reset_index()`. NEVER use `.reset()`!
- PUT ALL YOUR CODE IN ONE SINGLE ```python ... ``` BLOCK. Do not write any explanations outside the code block.
            """
            
            generated_code = ""
            try:
                # 1. Ask Gemini for code
                generated_code = generate_python_code(system_prompt, selected_model, user_api_key)
                
                # 2. Execute the code safely in the current context
                import matplotlib.pyplot as plt
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt}
                exec(generated_code, globals(), local_vars)
                
                # 3. Retrieve the 'result' variable
                if "result" in local_vars:
                    response = local_vars["result"]
                    if isinstance(response, pd.DataFrame) or isinstance(response, pd.Series):
                        st.dataframe(response)
                    else:
                        st.write(response)
                else:
                    st.error("The model executed the code, but did not declare a 'result' variable.")

                with st.expander("View generated Python code"):
                    st.code(generated_code, language="python")

            except Exception as e:
                st.error(f"Error during execution: {e}")
                with st.expander("View generated Python code"):
                    st.code(generated_code, language="python")
else:
    st.info("Please upload a CSV file in the sidebar to begin.")
