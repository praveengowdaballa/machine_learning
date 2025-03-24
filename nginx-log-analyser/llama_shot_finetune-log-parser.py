import argparse
import requests
import yaml
import logging
from dotenv import load_dotenv
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='log-analyzer.log',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def call_ollama(full_prompt, model, ip, port):
    OLLAMA_URL = f"http://{ip}:{port}/api/generate"
    data = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "temperature": 0.2,  # Slightly increase creativity for better summaries
        "max_tokens": 4096
    }
    headers = {"Content-Type": "application/json"}
    return requests.post(OLLAMA_URL, json=data, headers=headers, timeout=300)

def load_few_shot_examples(directory="shot-learning", filename="examples.txt"):
    """Loads Few-Shot Learning examples from a file."""
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""  # Return empty string if no examples found

def save_response_to_file(filename, log_filename, num_lines, system, model, answer):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"Datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log Filename: {log_filename}\n")
        f.write(f"Number of Lines Analyzed: {num_lines}\n")
        f.write(f"System: {system}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Answer:\n{answer}\n")
        f.write("="*60 + "\n\n")

def main():
    logging.info("# Starting Nginx log analyzer")
    print('Starting the Nginx log analyzer')

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Send a YAML-based prompt plus the last N lines from an Nginx log file to Ollama."
    )
    parser.add_argument('-f', required=True, help='Path to the Nginx log file')
    parser.add_argument('-c', required=True, help='Path to the YAML config file')
    parser.add_argument('-q', default='', help='User query to refine log analysis')
    parser.add_argument('-ip', default='localhost', help='IP address of the Ollama server')
    parser.add_argument('-p', default='11434', help='Ollama server port')
    parser.add_argument('-m', default='llama2', help='Model name for Ollama')
    parser.add_argument('-n', type=int, default=10, help='Number of lines to read from file')
    parser.add_argument('-o', default='log-analyzer-output.txt', help='Output file')
    args = parser.parse_args()

    logging.info("Reading file: %s", args.f)
    with open(args.f, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    last_n_lines = lines[-args.n:] if len(lines) > args.n else lines
    logging.debug("Last %d lines:\n%s", args.n, "\n".join(last_n_lines))

    logging.info("Reading config file: %s", args.c)
    with open(args.c, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    few_shot_examples = load_few_shot_examples()
    
    base_prompt = (
        "You are an advanced Nginx log analysis AI.\n"
        "Analyze the given log entries with the following steps:\n"
        "1. Identify errors, failed requests, or security threats.\n"
        "2. Categorize issues (Client Error, Server Error, DDoS Attempt, etc.).\n"
        "3. Extract timestamps, affected URLs, and IP addresses.\n"
        "4. Summarize findings in structured format.\n\n"
        "If applicable, highlight repeated failures or suspicious patterns."
    )
    
    full_prompt = (
        f"{few_shot_examples}\n\n"  # Prepend few-shot examples
        f"{base_prompt}\n\n"
        f"User Query: {args.q}\n\n" if args.q else f"{base_prompt}\n\n"
        + "\n".join(last_n_lines)
    )
    
    logging.debug("Full prompt:\n%s", full_prompt)
    print('-> Sending POST request to Ollama')
    response = call_ollama(full_prompt, args.m, args.ip, args.p)

    if response:
        logging.debug("Response status: %d", response.status_code)
        logging.debug("Response body:\n%s", response.text)
        if response.status_code == 200:
            resp_data = response.json()
            if "response" in resp_data:
                final_answer = resp_data["response"].strip()
            else:
                logging.warning("No valid response field found.")
                print("\nNo valid response field found in JSON.\n")
                return
        else:
            logging.error("Error: %d - %s", response.status_code, response.text)
            print(f"\nError: {response.status_code} - {response.text}\n")
            return

        logging.info("Received valid response.")
        print("\n" + "="*60)
        print("LLM RESPONSE:")
        print("="*60)
        print(final_answer)
        print("="*60 + "\n")

        save_response_to_file(args.o, args.f, args.n, "ollama", args.m, final_answer)
    else:
        logging.error("Failed to get a response from the server.")

    logging.info("Finished main function")

if __name__ == "__main__":
    main()
