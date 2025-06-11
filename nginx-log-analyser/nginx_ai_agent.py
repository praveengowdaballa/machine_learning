import os
import re
import psutil
import pandas as pd
import streamlit as st
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# =============================================
# SETUP & CONFIGURATION
# =============================================

# Configure Streamlit page
st.set_page_config(
    page_title="Nginx LogSight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        border-radius: 5px;
    }
    .stDateInput>div>div>input {
        border-radius: 5px;
    }
    .stExpander {
        border-left: 3px solid #3498db;
    }
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .logo {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# SIDEBAR CONFIGURATION
# =============================================

with st.sidebar:
    st.image("Opentextlogo.png", width=200)
    st.markdown('<div class="logo">Nginx LogSight Pro</div>', unsafe_allow_html=True)
    
    # Model Configuration
    with st.expander("⚙️ Model Settings", expanded=True):
        selected_model = st.selectbox(
            "LLM Model",
            ["llama3:latest", "llama2:latest", "gemma:latest", "mistral:latest"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
        max_tokens = st.number_input("Max Response Tokens", 100, 2000, 500)
    
    # Log Filters
    with st.expander("🔍 Filter Settings", expanded=True):
        keyword_filter = st.text_input("Keyword filter", "")
        regex_pattern = st.text_input("Regex pattern", "")
        start_date = st.date_input("Start Date", datetime.now().date())
        end_date = st.date_input("End Date", datetime.now().date())
        
        if start_date > end_date:
            st.error("End date must be after start date")
            st.stop()
    
    # File Management
    with st.expander("📁 Log Management", expanded=True):
        logs_path = "data/logs"
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
            st.success(f"Created directory `{logs_path}`")
        
        uploaded_file = st.file_uploader("Upload log file", type=["log", "txt"])
        if uploaded_file:
            try:
                file_path = os.path.join(logs_path, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"❌ Failed to upload file: {str(e)}")
    
    # System Monitoring
    with st.expander("📊 System Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
        with col2:
            st.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")

# =============================================
# MAIN INTERFACE
# =============================================

# Header with logo and description
col1, col2 = st.columns([1, 4])
with col1:
    st.image("Opentextlogo.png", width=300)
with col2:
    st.title("Nginx LogSight Pro")
    st.markdown("AI-powered log analysis platform for enterprise-grade Nginx log monitoring and troubleshooting")

st.markdown("---")

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# =============================================
# CORE FUNCTIONS
# =============================================

def preprocess_log_documents(docs):
    """Enhance log documents with metadata and parsing"""
    for doc in docs:
        log_entry = doc.page_content.strip()
        if log_entry:
            try:
                # Extract timestamp from common nginx log formats
                if '[' in log_entry and ']' in log_entry:
                    timestamp_str = log_entry.split('[')[1].split(']')[0]
                    dt = datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S %z")
                    doc.metadata["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                elif len(log_entry.split()) > 3:
                    # Try other common formats
                    dt = datetime.strptime(log_entry.split()[3][1:], "%d/%b/%Y:%H:%M:%S")
                    doc.metadata["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
    return docs

@st.cache_resource(show_spinner="🔄 Loading & Embedding Logs...")
def load_logs_and_create_vectorstore(path: str):
    """Load and process log documents"""
    loader = DirectoryLoader(path, glob="*.log", use_multithreading=True)
    documents = preprocess_log_documents(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "  ", " ", ""]
    )
    split_docs = splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_documents(split_docs, embeddings)

@st.cache_resource(show_spinner="🧠 Initializing Model...")
def get_llm(model_name: str):
    """Initialize LLM with configurable parameters"""
    return OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_ctx=max_tokens
    )

@st.cache_resource(show_spinner="🔗 Building Response Chain...")
def get_chain(_llm):
    """Create processing chain with enhanced prompt"""
    prompt = ChatPromptTemplate.from_template("""
    # Nginx Log Analysis Request
    
    ## Context
    You are LogSight AI, an expert system for Nginx log analysis with the following capabilities:
    - Deep understanding of Nginx log formats (combined, main, error)
    - Security threat detection
    - Performance bottleneck identification
    - Traffic pattern analysis
    
    ## Log Context
    {context}
    
    ## User Query
    {input}
    
    ## Response Requirements
    1. First analyze the overall log patterns
    2. Identify any anomalies or important events
    3. Specifically address the user query
    4. Highlight security concerns if any
    5. Provide timeline analysis if relevant
    6. Format professionally with Markdown
    7. Use code blocks for log examples
    
    ## Final Answer
    """)
    return create_stuff_documents_chain(llm=_llm, prompt=prompt)

def filter_doc(doc):
    """Apply all filters to documents"""
    content = doc.page_content.lower()
    
    # Keyword filter
    if keyword_filter and keyword_filter.lower() not in content:
        return False
    
    # Regex filter
    if regex_pattern:
        try:
            if not re.search(regex_pattern, doc.page_content, re.IGNORECASE):
                return False
        except re.error:
            st.warning("Invalid regex pattern - ignoring filter")
    
    # Date filter
    if "timestamp" in doc.metadata:
        try:
            dt = datetime.strptime(doc.metadata["timestamp"], "%Y-%m-%d %H:%M:%S")
            return start_date <= dt.date() <= end_date
        except Exception:
            return True  # ignore if timestamp parsing fails
    
    return True

# =============================================
# APPLICATION LOGIC
# =============================================

# Load data and models
try:
    vectorstore = load_logs_and_create_vectorstore(logs_path)
    llm = get_llm(selected_model)
    chain = get_chain(llm)
except Exception as e:
    st.error(f"Failed to initialize system: {str(e)}")
    st.stop()

# Main query interface
with st.container():
    query = st.text_input(
        "🔍 Enter your Nginx log analysis query:",
        placeholder="e.g. What errors occurred between 2-3am? Show examples and suggest fixes.",
        key="query_input"
    )

# Query processing
if query:
    with st.spinner("🔍 Analyzing logs..."):
        try:
            # Search and filter documents
            results = vectorstore.similarity_search(query, k=7)
            filtered_results = list(filter(filter_doc, results))
            
            if not filtered_results:
                st.warning("No logs matched your filters. Try broadening your search criteria.")
                st.stop()
            
            # Generate response
            response = chain.invoke({
                "context": filtered_results,
                "input": query
            })
            
            # Update conversation history
            st.session_state.conversation.append(("user", query))
            st.session_state.conversation.append(("agent", response))
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["AI Analysis", "Log Samples", "Statistics"])
            
            with tab1:
                st.markdown("### 📝 Analysis Report")
                st.markdown(response)
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Export as Markdown",
                        data=response,
                        file_name="log_analysis_report.md",
                        mime="text/markdown",
                    )
                with col2:
                    st.download_button(
                        label="📊 Export as CSV",
                        data=pd.DataFrame({"Analysis": [response]}).to_csv(index=False),
                        file_name="log_analysis.csv",
                        mime="text/csv",
                    )
            
            with tab2:
                st.markdown("### 📄 Relevant Log Samples")
                for i, doc in enumerate(filtered_results, 1):
                    filename = os.path.basename(doc.metadata.get("source", ""))
                    timestamp = doc.metadata.get("timestamp", "unknown")
                    
                    with st.expander(f"📝 Sample {i} - {filename} ({timestamp})"):
                        st.code(doc.page_content, language="log")
                        st.caption(f"Similarity score: {doc.metadata.get('score', 'N/A')}")
            
            with tab3:
                st.markdown("### 📈 Log Statistics")
                
                # Timeline visualization
                timestamps = []
                for doc in filtered_results:
                    if "timestamp" in doc.metadata:
                        try:
                            timestamps.append(datetime.strptime(doc.metadata["timestamp"], "%Y-%m-%d %H:%M:%S"))
                        except:
                            pass
                
                if timestamps:
                    st.markdown("#### ⏱️ Timeline Distribution")
                    df = pd.DataFrame({"timestamp": timestamps})
                    df['count'] = 1
                    df = df.set_index('timestamp').resample('h').sum().fillna(0)
                    st.area_chart(df)
                
                # Keyword frequency
                st.markdown("#### 🔤 Keyword Frequency")
                all_text = " ".join([doc.page_content for doc in filtered_results])
                words = re.findall(r'\b\w{4,}\b', all_text.lower())
                if words:
                    word_freq = pd.Series(words).value_counts().head(10)
                    st.bar_chart(word_freq)
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Conversation history
if st.session_state.conversation:
    st.markdown("---")
    st.markdown("### 📜 Conversation History")
    
    for i, (speaker, text) in enumerate(st.session_state.conversation):
        if speaker == "user":
            with st.chat_message("user"):
                st.markdown(text)
        else:
            with st.chat_message("assistant"):
                st.markdown(text)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
    <p>Nginx LogSight Pro • Built By SRE Core Opentext Powered by Ollama and LangChain</p>
    <p>© 2025 SRE Core Team • Enterprise-ready log analysis</p>
</div>
""", unsafe_allow_html=True)
