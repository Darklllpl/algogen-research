import os
import sys
# Define script directory constants, use absolute paths to avoid relative path issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default paths
PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "llm_creative_prompt.txt")
RESULT_FILE = os.path.join(BASE_DIR, "llm_result", "llm_creative_result.txt")
import requests
import json  # Add json for pretty printing debug

def read_prompt_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_result(content):
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    # Support command line specified prompt file path
    prompt_file = PROMPT_FILE
    if len(sys.argv) > 1:
        prompt_file = sys.argv[1]
    prompt = read_prompt_file(prompt_file)
    # Print prompt, show prompt file path being used
    print(f"----- Using prompt file ({prompt_file}) content to request LLM -----")
    if not prompt.strip():
        err = "[Error] Creative prompt file is empty!"
        write_result(err)
        print(err)
        sys.exit(1)
    # Replace LLM call with requests
    token = "sk-dMdaYSQFPlEMDhGK02AeD8C2Ec0d43EdBaD8Ce0435BcC623"
    url = 'https://az.gptplus5.com/v1/chat/completions'
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
    try:
        response = requests.post(url, headers=headers, json=data).json()
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            reasoning = ""
            try:
                content_obj = json.loads(content)
                result = content_obj
            except Exception:
                result = content  # fallback: if not JSON, write as is
            if not str(content).strip() or str(content).strip() == '{}':
                result = "[Error] API returned empty or invalid content (may be prompt issue)"
        else:
            result = "[Error] No valid response from API"
        os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            if isinstance(result, dict):
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                f.write(result)
        print(f"Written to llm_creative_result.txt")
    except Exception as e:
        print(f"[Error] Exception during LLM request or result writing: {e}")
        write_result(f"[Error] Exception during LLM request or result writing: {e}")
        sys.exit(1)