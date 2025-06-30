import os
from dotenv import load_dotenv
import requests
import json
import subprocess
import sys

def get_gemini_response(question):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": question}]}]}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.ok:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "[Gemini API Response Format Error]"
    else:
        return "[Gemini API Error]"

def get_serper_facts(question):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    data = {"q": question}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.ok:
        try:
            results = response.json()
            facts = [item["snippet"] for item in results.get("organic", []) if "snippet" in item]
            return " ".join(facts)
        except Exception:
            return "[Serper API Response Format Error]"
    else:
        return "[Serper API Error]"

def detect_hallucination(fact, response):
    try:
        result = subprocess.run([
            sys.executable, "inference.py", "--fact", fact, "--response", response
        ], capture_output=True, text=True, timeout=60)
        verdict = result.stdout.strip()
        if not verdict:
            verdict = f"[No verdict returned. Check inference.py output. Stderr: {result.stderr.strip()}]"
        return verdict
    except Exception as e:
        return f"[Error running inference.py: {e}]"

def browse_dataset():
    print("Choose dataset: 1) qa_data.json  2) general_data.json")
    choice = input("Enter 1 or 2: ")
    if choice == "1":
        with open("qa_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"There are {len(data)} questions in qa_data.json.")
        qidx = int(input(f"Select question number (1 to {len(data)}): ")) - 1
        question = data[qidx]["question"]
        fact = data[qidx]["knowledge"]
        response = data[qidx]["right_answer"]
    else:
        with open("general_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"There are {len(data)} questions in general_data.json.")
        qidx = int(input(f"Select question number (1 to {len(data)}): ")) - 1
        question = data[qidx]["user_query"]
        fact = get_serper_facts(question)
        response = data[qidx]["chatgpt_response"]
    return question, fact, response

# Load API keys from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

print("Choose input mode: 1) Real-time question  2) Browse dataset")
mode = input("Enter 1 or 2: ")
if mode == "1":
    question = input("Enter your real-time question: ")
    llm_response = get_gemini_response(question)
    facts = get_serper_facts(question)
    hallucination_result = detect_hallucination(facts, llm_response)
    print("\nQuestion:", question)
    print("Gemini Response:", llm_response)
    print("Supporting Facts:", facts)
    print("Hallucination Verdict:", hallucination_result)
else:
    question, fact, response = browse_dataset()
    hallucination_result = detect_hallucination(fact, response)
    print("\nQuestion:", question)
    print("Response:", response)
    print("Supporting Facts:", fact)
    print("Hallucination Verdict:", hallucination_result)
    choice = input("1) Run Hallucination Detection \n2) Return to the menu\n")
    if choice == '1':
        question = input("Enter your question: ")
        response = get_gemini_response(question)
        facts = get_serper_facts(question)
        verdict = detect_hallucination(facts, response)
        print(f"\nQuestion: {question}")
        print(f"Gemini Response: {response}")
        print(f"Supporting Facts: {facts}")
        print(f"Hallucination Verdict: {verdict}\n")
        input("Press Enter to return to the menu...")
    else:
        hallucination_result = detect_hallucination(fact, response)
        print("\nQuestion:", question)
        print("Response:", response)
        print("Supporting Facts:", fact)
        print("Hallucination Verdict:", hallucination_result)