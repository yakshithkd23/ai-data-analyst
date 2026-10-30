import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

# --- INITIALIZE STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

st.title(" AI-Powered Data Analyst")
st.markdown("Upload your CSVs and ask questions, generate charts, or detect anomalies.")

# --- SIDEBAR: CONFIG & UPLOAD ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Groq API Key", type="password")
    
    st.header("Data Upload")
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.dataframes = []
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                st.session_state.dataframes.append(df)
                st.success(f"Loaded {file.name} successfully!")
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")

# --- AGENT INITIALIZATION ---
def get_agent():
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return None
    if not st.session_state.dataframes:
        st.warning("Please upload at least one CSV file.")
        return None
    
    # llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=api_key)
    # Llama 3 70B is an excellent open-source reasoning model available on Groq
    llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b", groq_api_key=api_key)    # Create agent handling single or multiple dataframes
    agent = create_pandas_dataframe_agent(
        llm,
        st.session_state.dataframes if len(st.session_state.dataframes) > 1 else st.session_state.dataframes[0],
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True, # Required for execution in experimental Pandas agent
        handle_parsing_errors=True
    )
    return agent

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question (e.g., 'Show monthly sales trends' or 'Detect anomalies')"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    agent = get_agent()
    if agent:
        with st.chat_message("assistant"):
            st_callback = st.spinner("Analyzing data and generating reasoning...")
            try:
                # Custom prompt injection to enforce reasoning and chart generation if requested
                system_prompt = f"""
                You are an expert AI Data Analyst. 
                Answer the user's query: "{prompt}".
                If the user asks for a chart, write the python code using matplotlib or seaborn, save it to 'temp_chart.png', and explain the chart.
                If the user asks for anomalies, use statistical methods (like IQR or Z-score) to find them, explain your reasoning, and output the flagged data.
                If the user asks for SQL, generate the equivalent SQL query alongside the Pandas analysis.
                """
                response = agent.invoke(system_prompt)
                
                output = response["output"]
                st.markdown(output)
                
                # Render chart if the agent generated one
                if os.path.exists("temp_chart.png"):
                    st.image("temp_chart.png")
                    os.remove("temp_chart.png") # Cleanup
                
                st.session_state.messages.append({"role": "assistant", "content": output})
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
