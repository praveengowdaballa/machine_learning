"""
Nginx Log Analysis with LLaMA 2 and RAG Integration
curl -fsSL https://ollama.com/install.sh | sh
pip install pandas scikit-learn langchain ollama
python3 -m venv nginx_venv
source nginx_venv/bin/activate
sudo apt install python3.12-venv
source nginx_venv/bin/activate
python3 -m venv nginx_venv
source nginx_venv/bin/activate
pip install pandas scikit-learn langchain ollama
pip install langchain_community
ollama pull llama2
pip install -U langchain langchain_ollama
pip install pandas langchain ollama chromadb sentence-transformers
Description:
This Python script analyzes Nginx reverse proxy logs using LLaMA 2 with a Retrieval-Augmented Generation (RAG) approach. It vectorizes log entries using ChromaDB and SentenceTransformers to enable intelligent querying for error patterns, issues, and mitigation steps.

Key Features:

Utilizes LLaMA 2 for expert-level Nginx log analysis.

Implements ChromaDB for efficient vector storage and similarity search.

Uses SentenceTransformer for text embedding to enhance search accuracy.

Supports dynamic knowledge base building from logs stored in the ./knowledge_base directory.

Provides actionable insights, including detailed mitigation steps for errors like '502 Bad Gateway', 'Host Not Found', and upstream server issues.

Dependencies:

pandas

langchain_ollama

chromadb

sentence-transformers

Usage:

Place your log files in the ./knowledge_base directory.

Run the script to build the knowledge base.

Ask relevant queries like:

"What URLs faced 'Host Not Found' issues?"

"Provide steps to resolve 502 errors."

"""

import os
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from chromadb import Client as ChromaClient
from sentence_transformers import SentenceTransformer

# Initialize FAISS and LLaMA 2
llm = OllamaLLM(model="llama2", stop=["<|eot_id|>"])
chroma_client = ChromaClient()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Directory for knowledge base logs
KNOWLEDGE_BASE_DIR = "./knowledge_base"

# Create or get the collection
db_collection = chroma_client.create_collection(name="nginx_logs")

# Load and vectorize log data
def build_knowledge_base():
    logs = []
    file_count = 0
    for file in os.listdir(KNOWLEDGE_BASE_DIR):
        with open(os.path.join(KNOWLEDGE_BASE_DIR, file), 'r') as f:
            new_logs = f.readlines()
            logs.extend(new_logs)
            file_count += 1

            # Vectorize newly added logs immediately
            embeddings = embedder.encode(new_logs, convert_to_tensor=True)
            for i, log in enumerate(new_logs):
                db_collection.add(
                    documents=[log],
                    embeddings=[embeddings[i].tolist()],
                    ids=[str(len(logs) + i)]  # Ensure unique IDs for new entries
                )
    print(f"✅ Successfully vectorized {file_count} files with {len(logs)} total entries.")

# Query LLaMA 2 with RAG integration
def query_llama2(query):
    # Retrieve relevant log data from ChromaDB
    query_embedding = embedder.encode([query])[0].tolist()
    results = db_collection.query(query_embeddings=[query_embedding], n_results=5)
    relevant_logs = "\n".join(results['documents'][0])

    # Construct prompt for enhanced LLaMA response
    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an Nginx log expert who provides detailed analysis and mitigation steps for issues found in logs.
        
        Relevant log data:
        {log_data}
        
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {query}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    prompt = PromptTemplate(input_variables=["query", "log_data"], template=template)
    response = llm.invoke(prompt.format(query=query, log_data=relevant_logs))
    return response

if __name__ == "__main__":
    build_knowledge_base()  # Build vectorized knowledge base
    query = input("Ask your Nginx expert query: ")
    print("Nginx Expert Response:", query_llama2(query))

