import streamlit as st
import time
from src.agents import create_agent

# --- Page Config ---
st.set_page_config(
    page_title="RAG Financial Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Styling ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #161b22;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #30363d;
    }
    
    /* User Message Override */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1f2428;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #238636;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #161b22;
        color: white;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    /* Header */
    h1 {
        background: linear-gradient(90deg, #79c0ff 0%, #d2a8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "AMD"

if "agent_instance" not in st.session_state:
    st.session_state.agent_instance = create_agent("AMD")

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Financial Analyst")
    st.markdown("---")
    
    # Agent Selection
    new_agent = st.selectbox(
        "Select Company Agent",
        ["AMD", "Intel", "NVIDIA"],
        index=["AMD", "Intel", "NVIDIA"].index(st.session_state.selected_agent)
    )
    
    # Handle Agent Switch
    if new_agent != st.session_state.selected_agent:
        st.session_state.selected_agent = new_agent
        st.session_state.agent_instance = create_agent(new_agent)
        st.session_state.messages = [] # Clear chat on switch to avoid context confusion
        st.rerun()
    
    st.markdown("### Agent Capabilities")
    st.info(f"""
    **Current Agent:** {new_agent}
    
    Analyze 10-K financial reports with RAG technology.
    """)
    
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.agent_instance.clear_memory()
        st.rerun()

# --- Main Chat Interface ---
st.title(f"{st.session_state.selected_agent} Financial Assistant")
st.markdown("Ask detailed questions about financial reports, revenue, and growth.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about financial data..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate Assistant Response
    with st.chat_message("assistant"):
        # Use stream method from agent
        response_stream = st.session_state.agent_instance.stream(prompt)
        response = st.write_stream(response_stream)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
