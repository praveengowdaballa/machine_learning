import streamlit as st
import subprocess
import hashlib
import json
import psutil
import os
import re
import time
import csv
from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import ollama
from streamlit_extras.stylable_container import stylable_container

# Import your cross-account assumption function
import boto3
import sys

def assume_cross_account_role(account_id, role_name, region, session_name='AssumeRoleSession1'):
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    
    # Use the EC2 instance's default role to call STS (or your default AWS credentials)
    sts_client = boto3.client('sts', region_name=region)

    # Assume the target role
    response = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name
    )

    credentials = response['Credentials']

    # We don't need to create a new session here if we are passing env vars
    # But it's good for direct use in the current process
    session = boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
        region_name=region
    )

    return credentials # Return credentials to pass as env vars

# =============================================
# SETUP & CONFIGURATION
# =============================================

# Configure Streamlit page
st.set_page_config(
    #page_title="🔒 WAF AI Assistant Pro",
    page_title="WAF AI Assistant Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auth Data Path
user_data_path = Path("data/users.json")
logs_path = Path("data/logs")
user_data_path.parent.mkdir(parents=True, exist_ok=True)
logs_path.mkdir(parents=True, exist_ok=True)
if not user_data_path.exists():
    user_data_path.write_text(json.dumps({"users": {}, "pending": {}}))

DOCS_PATH = "docs/waf"
MODEL_NAME = "llama3:latest"

# =============================================
# AUTHENTICATION SYSTEM
# =============================================

# Auth Helpers
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()
def load_user_db(): return json.load(open(user_data_path))
def save_user_db(d): json.dump(d, open(user_data_path, "w"), indent=2)

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# Custom CSS for modern styling
st.markdown("""
<style>
    .auth-container {
        max-width: 480px;
        margin: 5rem auto;
        padding: 3rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        position: relative;
        overflow: hidden;
    }
    
    .auth-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, 
            rgba(74, 108, 247, 0.08) 0%, 
            rgba(255, 255, 255, 0) 50%, 
            rgba(37, 65, 178, 0.08) 100%);
        z-index: -1;
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .logo-text {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4a6cf7 0%, #2541b2 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .logo-subtext {
        color: #6c757d;
        margin-bottom: 2.5rem;
        text-align: center;
        font-size: 0.95rem;
    }
    
    .stTextInput input {
        height: 48px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 0 16px;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #4a6cf7;
        box-shadow: 0 0 0 3px rgba(74, 108, 247, 0.15);
    }
    
    .stButton>button {
        width: 100%;
        height: 48px;
        border-radius: 10px;
        background: linear-gradient(90deg, #4a6cf7 0%, #2541b2 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        font-size: 1rem;
        margin-top: 8px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(74, 108, 247, 0.25);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        margin: 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        background: rgba(74, 108, 247, 0.08);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4a6cf7 0%, #2541b2 100%);
        color: white !important;
    }
    
    .divider {
        display: flex;
        align-items: center;
        margin: 1.5rem 0;
        color: #6c757d;
        font-size: 0.8rem;
    }
    
    .divider::before, .divider::after {
        content: "";
        flex: 1;
        border-bottom: 1px solid #e0e0e0;
        margin: 0 10px;
    }
    
    /* Main app styling */
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
    .command-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #3498db;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Logout
if st.session_state.auth_user:
    with st.sidebar:
        if st.button("🚪 Logout"):
            st.session_state.auth_user = None
            st.success("Logged out successfully.")
            st.rerun()

# Login Page
if st.session_state.auth_user is None:
    # Auth container
    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            with stylable_container(
                key="auth_box",
                css_styles="""
                    {
                        padding: 2rem;
                        border-radius: 12px;
                    }
                """,
            ):
                st.markdown('<div class="logo-text">🛡️WAF AI Assistant</div>', unsafe_allow_html=True)
                st.markdown('<div class="logo-subtext">Enterprise WAF Management Platform</div>', unsafe_allow_html=True)
                
                tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
                db = load_user_db()

                with tab_login:
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="••••••••")
                    
                    if st.button("Login", key="login_btn"):
                        if username == "sre-core-admin" and password == "cc0e4help1!":
                            st.session_state.auth_user = "sre-core-admin"
                            st.success("✅ Logged in as admin")
                            st.rerun()
                        elif username in db['users'] and db['users'][username]['password'] == hash_password(password):
                            st.session_state.auth_user = username
                            st.success(f"✅ Logged in as {username}")
                            st.rerun()
                        elif username in db['pending']:
                            st.warning("🕒 Your account is pending admin approval")
                        else:
                            st.error("❌ Invalid username or password")
                
                with tab_signup:
                    new_user = st.text_input("Corporate Email", placeholder="john@opentext.com")
                    new_pass = st.text_input("New Password", type="password", placeholder="Create a strong password")
                    
                    if st.button("Request Access", key="signup_btn"):
                        if not new_user.endswith("@opentext.com"):
                            st.warning("❌ Only @opentext.com emails allowed")
                        elif new_user in db['users'] or new_user in db['pending']:
                            st.warning("ℹ️ Account already exists or pending approval")
                        else:
                            db['pending'][new_user] = {"password": hash_password(new_pass)}
                            save_user_db(db)
                            st.success("✅ Request submitted! You'll receive an email when approved.")

    st.stop()

# Admin Panel for Approvals
if st.session_state.auth_user == "sre-core-admin":
    st.sidebar.markdown("### 🛠 Admin Panel")
    db = load_user_db()
    if db['pending']:
        for user, data in list(db['pending'].items()):
            with st.sidebar.expander(f"👤 {user}"):
                if st.button(f"✅ Approve {user}", key=f"approve_{user}"):
                    db['users'][user] = data
                    del db['pending'][user]
                    save_user_db(db)
                    st.success(f"{user} approved")
                    st.rerun()
                if st.button(f"❌ Reject {user}", key=f"reject_{user}"):
                    del db['pending'][user]
                    save_user_db(db)
                    st.warning(f"{user} rejected")
                    st.rerun()
    else:
        st.sidebar.info("No pending users.")

# Block access if not approved user
if st.session_state.auth_user != "sre-core-admin" and st.session_state.auth_user not in load_user_db().get("users", {}):
    st.error("Access Denied")
    st.stop()

# =============================================
# LOGGING SYSTEM
# =============================================

def log_action(user, action_type, prompt, response=None, command=None, output=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "user": user,
        "action_type": action_type,
        "prompt": prompt,
        "response": response,
        "command": command,
        "output": output
    }
    
    # Save to individual user log
    user_log_path = logs_path / f"{user}.csv"
    file_exists = user_log_path.exists()
    
    with open(user_log_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)
    
    # Also save to main log
    main_log_path = logs_path / "all_actions.csv"
    file_exists = main_log_path.exists()
    
    with open(main_log_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

# =============================================
# DOCUMENT PROCESSING
# =============================================

@st.cache_resource(show_spinner="🔄 Indexing WAF docs...")
def build_vectorstore():
    loader = DirectoryLoader(DOCS_PATH, glob="*.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embed = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_documents(chunks, embed)

@st.cache_resource(show_spinner="🧠 Loading LLM...")
def get_llm():
    return OllamaLLM(model=MODEL_NAME)

@st.cache_resource(show_spinner="🔗 Initializing Q&A Chain...")
def get_qa_chain(_llm):
    prompt = ChatPromptTemplate.from_template("""
    ## AWS WAF Q&A

    ### Context
    {context}

    ### Question
    {input}

    ### Answer
    Provide a professional explanation, referencing AWS WAF concepts. Include logics, use cases, and examples if relevant.
    """)
    return create_stuff_documents_chain(llm=_llm, prompt=prompt)


WAF_MIGRATION_SYSTEM_PROMPT = """
You are an AWS WAF Classic to WAFv2 migration assistant. 
You ONLY handle requests related to migrating WAF Classic WebACLs to WAFv2.
You are aware that the migration CLI script is located at 'scripts/migrate.py'.

Here are the valid commands you can output (choose one and return ONLY the command, no explanation):

1. To migrate a specific WebACL by name:
   python3 scripts/migrate.py --web-acl-name <WEBACL_NAME>

2. To migrate all WebACLs:
   python3 scripts/migrate.py --migrate-all

3. To list all WebACLs:
   python3 scripts/migrate.py --list

4. To migrate in batch:
   python3 scripts/migrate.py --migrate-batch

Rules:
- Extract WebACL names from user input when needed
- Always use the exact command formats shown above
- Never generate invalid or unrelated commands
- Do NOT return explanation, only the raw command string
- If the request is NOT about WAF migration, respond with:
  I can only help with AWS WAF Classic to WAFv2 migration requests.
- No explanation of bullet points just a command
"""



RESPONSE_TEMPLATES = {
    "list": "I've retrieved the list of all your WebACLs in AWS WAF Classic.",
    "migrate_single": "I've initiated the migration of WebACL '{name}' from Classic to WAFv2.",
    "migrate_batch": "I've started batch migration for the specified WebACLs to WAFv2.",
    "migrate_all": "I've begun migrating all WebACLs from Classic to WAFv2.",
    "success": "✅ Operation completed successfully. Here are the details:",
    "error": "❌ The operation encountered an error:",
    "confirmation": {
        "list": "Should I proceed with listing all WebACLs?",
        "migrate_single": "Ready to migrate WebACL '{name}' to WAFv2?",
        "migrate_batch": "Ready to migrate these WebACLs to WAFv2?",
        "migrate_all": "Ready to migrate all WebACLs to WAFv2?"
    }
}


def get_waf_command(user_input):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": WAF_MIGRATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )["message"]["content"].strip()
        
        # Enhanced command validation
        if not response:
            st.error("Empty response from LLM")
            return None, None, None
            
        if response.startswith("python3 scripts/migrate.py --list"):
            return response, "list", None
        elif "--web-acl-name" in response:
            try:
                name = re.search(r"--web-acl-name\s+([^\s]+)", response).group(1)
                return response, "migrate_single", name
            except AttributeError:
                st.error("Could not extract WebACL name from command")
                return None, None, None
        elif "--migrate-batch" in response:
            return response, "migrate_batch", None
        elif "--migrate-all" in response:
            return response, "migrate_all", None
        elif "I can only help with AWS WAF" in response:
            return None, None, None
        else:
            st.error(f"Invalid command format: {response}")
            return None, None, None
            
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        return None, None, None



# ==================================
# Action Handlers (These are mock functions, actual execution happens via subprocess)
# ==================================

def waf_list(mode):
    st.write(f"[DEBUG] WAF list fetched. Mode: {mode}")
    st.success("✅ Displaying list of WAFs (mocked)")


def migrate_single(query, mode):
    st.write(f"[DEBUG] Migrating single WAF. Mode: {mode}")
    st.success("✅ Migrated single WAF (mocked)")


def migrate_batch(query, mode):
    st.write(f"[DEBUG] Migrating batch WAFs. Mode: {mode}")
    st.success("✅ Migrated batch WAFs (mocked)")


def migrate_all(mode):
    st.write(f"[DEBUG] Migrating all WAFs. Mode: {mode}")
    st.success("✅ All WAFs migrated (mocked)")


def explain_intent(query):
    st.info(f"🧠 Explaining: {query}")
    st.markdown("""
    ### 🔍 WAF Migration Explained
    - **waf list**: Lists all WAFs in your AWS account
    - **migrate single**: Migrates a specific WAF from v1 to v2
    - **migrate batch**: Migrates a list of specified WAFs
    - **migrate all**: Migrates every available WAF
    - **explain**: Provides information about the commands
    """)


def clean_prompt(text):
    """Basic prompt cleaning"""
    return re.sub(r'^(hi|hello|please)[\s,]*', '', text.strip(), flags=re.IGNORECASE)



# =============================================
# CORE FUNCTIONS
# =============================================

def extract_valid_command(text):
    """Extract migration commands from text"""
    for line in text.splitlines():
        if re.match(r"^python3 scripts/migrate.py --\w+", line.strip()):
            return line.strip()
    return None

def generate_notification(intent, cmd=None):
    """Generate natural language notifications"""
    if intent == "list":
        return "✅ Retrieved list of WebACLs successfully"
    elif intent.startswith("migrate"): 
        if "--web-acl-name" in cmd:
            acl_name_match = re.search(r"--web-acl-name\s+([\w-]+)", cmd)
            if acl_name_match:
                acl_name = acl_name_match.group(1)
                return f"✅ Successfully initiated migration for WebACL '{acl_name}'"
            return "✅ Successfully initiated migration for a specific WebACL"
        elif "--migrate-batch" in cmd:
            return "✅ Successfully initiated batch migration for WebACLs"
        elif "--migrate-all" in cmd:
            return "✅ Successfully initiated migration for all WebACLs"
        return "✅ Migration command executed successfully"
    return None

# =============================================
# SIDEBAR CONFIGURATION
# =============================================

with st.sidebar:
    st.image("branding/Opentextlogo.png", width=200)
    st.markdown('<div class="logo">AWS WAF AI Assistant</div>', unsafe_allow_html=True)
    
    # AWS Account Configuration
    with st.expander("☁️ AWS Account Settings", expanded=True):
        st.session_state.aws_account_id = st.text_input(
            "Target AWS Account ID",
            value=st.session_state.get("aws_account_id", ""),
            help="The 12-digit AWS account ID where WAF operations will be performed."
        )
        st.session_state.aws_role_name = st.text_input(
            "IAM Role Name",
            value=st.session_state.get("aws_role_name", ""),
            help="The name of the IAM role to assume in the target account (e.g., 'WAFMigrationRole')."
        )
        st.session_state.aws_region = st.text_input(
            "AWS Region",
            value=st.session_state.get("aws_region", "us-east-1"),
            help="The AWS region for WAF operations (e.g., 'us-east-1', 'eu-west-1')."
        )

    # Model Configuration
    with st.expander("⚙️ Model Settings", expanded=True):
        selected_model = st.selectbox(
            "LLM Model",
            ["llama3:latest", "llama2:latest"],
            index=0,
            key="model_select_sidebar" 
        )
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, key="temp_slider_sidebar") 
    
    # System Monitoring
    with st.expander("📊 System Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
        with col2:
            st.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
    
    # Log Download
    with st.expander("📁 Activity Logs", expanded=True):
        user_log_path = logs_path / f"{st.session_state.auth_user}.csv"
        if user_log_path.exists():
            with open(user_log_path, "rb") as f:
                st.download_button(
                    label="📥 Download My Logs",
                    data=f,
                    file_name=f"waf_activity_{st.session_state.auth_user}.csv",
                    mime="text/csv",
                    key="download_button_sidebar"
                )
        else:
            st.info("No activity logs yet")

# =============================================
# MAIN INTERFACE
# =============================================
def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    # Main Interface
    st.title("AWS WAF AI Assistant Pro")
    st.markdown("AI-powered management platform for AWS WAF migration and configuration")
    st.markdown("---")

    # User input
    user_input = st.text_input(
        "🔍 Enter WAF migration request:",
        placeholder="e.g. 'migrate all webacls', 'list webacls', 'help me migrate waf named myacl'",
        key="waf_query"
    )

    if user_input:
        with st.spinner("🧠 Processing your request..."):
            command, action_type, name = get_waf_command(user_input)
            if not command:
                st.warning("I can only help with AWS WAF Classic to WAFv2 migration requests.")
                return
            
            if command:
                # Display natural language interpretation
                
                if action_type == "list":
                    st.info("🔍 " + RESPONSE_TEMPLATES["confirmation"]["list"])
                elif action_type == "migrate_single":
                    st.info("🚀 " + RESPONSE_TEMPLATES["confirmation"]["migrate_single"].format(name=name))
                elif action_type == "migrate_batch":
                    st.info("🔄 " + RESPONSE_TEMPLATES["confirmation"]["migrate_batch"])
                elif action_type == "migrate_all":
                    st.info("🌐 " + RESPONSE_TEMPLATES["confirmation"]["migrate_all"])
                st.markdown("### Command Preview")
                st.code(command, language="bash")
                
                # Two-column layout for approve/cancel
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve & Run", type="primary"):
                        # Validate AWS inputs
                        account_id = st.session_state.get("aws_account_id")
                        role_name = st.session_state.get("aws_role_name")
                        region = st.session_state.get("aws_region")

                        if not (account_id and role_name and region):
                            st.error("Please provide AWS Account ID, IAM Role Name, and Region in the sidebar before running commands.")
                            return

                        with st.spinner("⚡ Executing command..."):
                            try:
                                assumed_credentials = assume_cross_account_role(account_id, role_name, region)
                                os.environ['AWS_ACCESS_KEY_ID'] = assumed_credentials['AccessKeyId']
                                os.environ['AWS_SECRET_ACCESS_KEY'] = assumed_credentials['SecretAccessKey']
                                os.environ['AWS_SESSION_TOKEN'] = assumed_credentials['SessionToken']
                                os.environ['AWS_DEFAULT_REGION'] = region
                                env_vars = os.environ.copy()

                                output = subprocess.check_output(
                                    command, 
                                    shell=True, 
                                    stderr=subprocess.STDOUT, 
                                    text=True,
                                    env=env_vars # Pass the modified environment
                                )
                                
                                # Natural language success response
                                st.success(RESPONSE_TEMPLATES["success"])
                                if action_type == "list":
                                    st.info(RESPONSE_TEMPLATES["list"])
                                elif action_type == "migrate_single":
                                    st.info(RESPONSE_TEMPLATES["migrate_single"].format(name=name))
                                elif action_type == "migrate_batch":
                                    st.info(RESPONSE_TEMPLATES["migrate_batch"])
                                elif action_type == "migrate_all":
                                    st.info(RESPONSE_TEMPLATES["migrate_all"])
                                
                                st.code(output, language="text")
                                log_action(st.session_state.auth_user, 
                                         action_type, 
                                         user_input, 
                                         command=command, 
                                         output=output)
                                
                            except subprocess.CalledProcessError as e:
                                st.error(RESPONSE_TEMPLATES["error"])
                                st.code(e.output, language="text")
                                log_action(st.session_state.auth_user,
                                         "command_failed",
                                         user_input,
                                         command=command,
                                         output=e.output)
                            except Exception as e:
                                st.error(f"An error occurred during role assumption or execution: {e}")
                                log_action(st.session_state.auth_user,
                                         "assumption_failed",
                                         user_input,
                                         command=command,
                                         output=str(e))
                
                with col2:
                    if st.button("❌ Cancel"):
                        st.warning("Operation cancelled")
                        log_action(st.session_state.auth_user,
                                 "operation_cancelled",
                                 user_input,
                                 command=command)
            
            else:
                st.warning("I can only help with AWS WAF Classic to WAFv2 migration requests.")
                log_action(st.session_state.auth_user,
                         "invalid_request",
                         user_input)

    # Add conversational history
    if st.session_state.conversation:
        st.markdown("---")
        st.subheader("Conversation History")
        for msg in st.session_state.conversation[-3:]:  # Show last 3 messages
            with st.chat_message("user"):
                st.markdown(f"**👤 User [{msg['timestamp']}]:** {msg['user']}")
            with st.chat_message("assistant"):
                if msg['bot']:  # Only show if there's a bot response
                    st.markdown(f"**🤖 Assistant [{msg['timestamp']}]:** {msg['bot']}")
    
    if user_input and command:
        st.session_state.conversation.append({
            "user": user_input,
            "bot": RESPONSE_TEMPLATES.get(action_type, "").format(name=name) if name else RESPONSE_TEMPLATES.get(action_type, ""),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    if st.session_state.conversation:
        st.markdown("---")
        st.subheader("Conversation History")
        for i, msg in enumerate(st.session_state.conversation[-3:]):  # Show last 3 messages
            with st.chat_message("user"):
                st.write(f"👤 [{msg['timestamp']}]: {msg['user']}")
            with st.chat_message("assistant"):
                st.write(f"🤖 [{msg['timestamp']}]: {msg['bot']}")
if __name__ == "__main__":
    if "auth_user" in st.session_state and st.session_state.auth_user:
        #st.session_state.auth_user = "admin" #If you only want the input prompt for testing without login logic, you can temporarily run:
        main()
    else:
        st.warning("Please login to use the WAF Assistant.")
