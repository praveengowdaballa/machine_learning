import re
import json
import argparse
import subprocess
from langchain_ollama import OllamaLLM

# Initialize LLM with the correct model
llm = OllamaLLM(model="llama3:latest")

def parse_natural_language(input_text):
    """
    Parse the input natural language into structured JSON using LLM.
    """
    prompt = f"""
    Act as a JSON formatter. Parse the following input:
    '{input_text}'
    Always output JSON with the keys 'account_id' and 'region'.
    Example: {{
        "account_id": "123456789012",
        "region": "us-west-2"
    }}
    """
    try:
        response = llm.invoke(prompt)
        print(f"Raw response from LLM: {response}")  # Debug raw response

        # Extract JSON using regex
        match = re.search(r"\{.*?\}", response, re.DOTALL)  # Matches the first JSON block
        if not match:
            raise ValueError("No valid JSON found in LLM response.")

        # Parse the extracted JSON string
        json_str = match.group(0)
        parsed_response = json.loads(json_str)

        # Validate keys
        if all(key in parsed_response for key in ["account_id", "region"]):
            return parsed_response
        else:
            raise ValueError("Parsed response does not contain all required keys.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def execute_python_script(script_name, account_id, region):
    """
    Executes the target Python script with the parsed arguments.
    """
    try:
        result = subprocess.run(
            ["python3", script_name, account_id, region],
            capture_output=True,
            text=True
        )
        print("=== Script Output ===")
        print(result.stdout)
        if result.stderr:
            print("=== Script Errors ===")
            print(result.stderr)
    except Exception as e:
        print(f"Error executing script {script_name}: {e}")

if __name__ == "__main__":
    # Use argparse to accept user arguments
    parser = argparse.ArgumentParser(description="Parse natural language input for EC2 deployment.")
    parser.add_argument("input_text", type=str, help="Natural language input (e.g., 'Deploy EC2 in Account 264153888999 us-west-2 region')")

    args = parser.parse_args()

    # Use the input_text argument
    parsed_arguments = parse_natural_language(args.input_text)
    if parsed_arguments:
        print("Parsed arguments:", parsed_arguments)

        # Extract arguments
        account_id = parsed_arguments["account_id"]
        region = parsed_arguments["region"]

        # Execute the ec2deploy.py script
        execute_python_script("ec2deploy.py", account_id, region)
