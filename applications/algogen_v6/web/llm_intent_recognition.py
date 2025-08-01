import os
import sys
import requests

# Define script directory constants, use absolute paths to avoid relative path issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "llm_intent_recognition_prompt.txt")
RESULT_FILE = os.path.join(BASE_DIR, "llm_result", "llm_intent_recognition_result.txt")

def read_prompt():
    if not os.path.exists(PROMPT_FILE):
        return ""
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

def write_result(content):
    # Ensure result directory exists
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    # Support command line arguments to specify prompt file
    prompt_file = PROMPT_FILE
    if len(sys.argv) > 1:
        prompt_file = sys.argv[1]
    def read_prompt_file(path):
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    prompt = read_prompt_file(prompt_file)
    if not prompt.strip():
        print("Prompt file is empty!")
        write_result("[Error] Prompt file is empty!")
        exit(1)
    print(f"----- Using prompt file ({prompt_file}) content to request LLM -----")
    # Replace LLM call with requests
    token = ""
    url = ''
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        #gpt-4.1-2025-04-14
        "model": "gpt-4.1-2025-04-14",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data).json()
    if "choices" in response and len(response["choices"]) > 0:
        content = response["choices"][0]["message"]["content"]
        # Assuming no reasoning in this API, set to empty
        reasoning = ""
        result = ""
        if reasoning:
            result += f"[Reasoning]\n{reasoning}\n"
        result += content
    else:
        result = "[Error] No valid response from API"
    write_result(result)
    print("Written to llm_intent_recognition_result.txt")