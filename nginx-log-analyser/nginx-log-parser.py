import argparse
import requests
import yaml
import logging
from dotenv import load_dotenv
import os
from datetime import datetime

# Configure logging to go to a file, not the console
logging.basicConfig(
    filename='log-analyzer.log',
    filemode='a',  # Use 'w' if you want to overwrite on each run
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
        "temperature": 0,
        "max_tokens": 4096  # Assuming 4096 is the max tokens for Ollama
    }
    headers = {"Content-Type": "application/json"}
    return requests.post(OLLAMA_URL, json=data, headers=headers, timeout=300)

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

    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Send a YAML-based prompt plus the last N lines from an Nginx log file "
            "to Ollama."
        )
    )
    parser.add_argument(
        '-f',
        required=True,
        help='Path to the Nginx log file'
    )
    parser.add_argument(
        '-c',
        required=True,
        help='Path to the YAML config file'
    )
    parser.add_argument(
        '-ip',
        default='localhost',
        help='IP address of the Ollama server (default: localhost)'
    )
    parser.add_argument(
        '-p',
        default='11434',
        help='Ollama server port (default: 11434)'
    )
    parser.add_argument(
        '-m',
        default='llama2',
        help='Model name for Ollama (default: llama2)'
    )
    parser.add_argument(
        '-n',
        type=int,
        default=10,
        help='Number of lines to read from file (default: 10)'
    )
    parser.add_argument(
        '-o',
        default='log-analyzer-output.txt',
        help='Output file to save the response (default: log-analyzer-output.txt)'
    )
    args = parser.parse_args()

    logging.info(
        "Parsed arguments: file=%s, config=%s, ip=%s, port=%s, model=%s, lines=%d, output=%s",
        args.f, args.c, args.ip, args.p, args.m, args.n, args.o
    )

    # Read the last N lines from the Nginx log file
    logging.info("Reading file: %s", args.f)
    with open(args.f, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if len(lines) > args.n:
        last_n_lines = lines[-args.n:]
    else:
        last_n_lines = lines

    logging.debug("Last %d lines from file:\n%s",
                  args.n, "\n".join(last_n_lines))

    # Read the YAML config (prompt)
    logging.info("Reading config file: %s", args.c)
    with open(args.c, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Update prompt for Nginx log analysis
    prompt = (
        "You are an Nginx log analysis expert. Analyze the following log entries "
        "and identify any anomalies, errors, security threats, or suspicious activity. "
        "Provide a summary of your findings without recommendations."
    )
    logging.debug("Prompt:\n%s", prompt)

    # Combine prompt with the last N lines
    full_prompt = f"{prompt}\n\n" + "\n".join(last_n_lines)
    logging.debug("Full prompt to send:\n%s", full_prompt)

    print(f'-> Sending POST request to Ollama')
    logging.info("Sending POST request to Ollama")
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
                print("\nNo valid response field found in the JSON.\n")
                return
        else:
            logging.error(
                "Error: %d - %s", response.status_code, response.text
            )
            print(f"\nError: {response.status_code} - {response.text}\n")
            return

        logging.info("Received valid response.")
        print("\n" + "="*60)
        print("LLM RESPONSE:")
        print("="*60)
        print(final_answer)
        print("="*60 + "\n")

        # Save the response to a file
        save_response_to_file(args.o, args.f, args.n, "ollama", args.m, final_answer)
    else:
        logging.error("Failed to get a response from the server.")

    logging.info("Finished main function")

if __name__ == "__main__":
    main()

