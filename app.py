import streamlit as st
import os
import json
import subprocess
import sys
from dotenv import load_dotenv
import requests


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

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

st.markdown("""
    <style>
    .big-title {color: #4F8BF9; font-size: 2.5em; font-weight: bold;}
    .section-header {color: #FF6F61; font-size: 1.3em; font-weight: bold; margin-top: 1em;}
    .verdict-box {padding: 1em; border-radius: 10px; font-size: 1.2em; margin-top: 1em;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="big-title">🤖 LLM Hallucination Detection</div>', unsafe_allow_html=True)
mode = st.radio("Choose input mode:", ("Real-time question", "Browse dataset"))

if mode == "Real-time question":
    question = st.text_input("Enter your question:")
    if st.button("Get Response and Verdict") and question:
        llm_response = get_gemini_response(question)
        facts = get_serper_facts(question)
        hallucination_result = detect_hallucination(facts, llm_response)
        st.markdown('<div class="section-header">Question</div>', unsafe_allow_html=True)
        st.info(question)
        st.markdown('<div class="section-header">Gemini Response</div>', unsafe_allow_html=True)
        st.success(llm_response)
        st.markdown('<div class="section-header">Supporting Facts</div>', unsafe_allow_html=True)
        st.warning(facts)
        st.markdown('<div class="section-header">Hallucination Verdict</div>', unsafe_allow_html=True)
        if "Not Hallucinated" in hallucination_result:
            st.markdown(f'<div class="verdict-box" style="background-color:#D4EDDA;color:#155724;">✅ {hallucination_result}</div>', unsafe_allow_html=True)
        elif "Hallucinated" in hallucination_result:
            st.markdown(f'<div class="verdict-box" style="background-color:#F8D7DA;color:#721C24;">❌ {hallucination_result}</div>', unsafe_allow_html=True)
        else:
            st.error(hallucination_result)
else:
    dataset = st.selectbox("Choose dataset:", ("qa_data.json", "general_data.json"))
    with open(dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    qidx = st.number_input(f"Select question number (1 to {len(data)}):", min_value=1, max_value=len(data), step=1) - 1
    if dataset == "qa_data.json":
        question = data[qidx]["question"]
        fact = data[qidx]["knowledge"]
        response = data[qidx]["right_answer"]
    else:
        question = data[qidx]["user_query"]
        fact = get_serper_facts(question)
        response = data[qidx]["chatgpt_response"]
    st.markdown('<div class="section-header">Question</div>', unsafe_allow_html=True)
    st.info(question)
    st.markdown('<div class="section-header">Response</div>', unsafe_allow_html=True)
    st.success(response)
    st.markdown('<div class="section-header">Supporting Facts</div>', unsafe_allow_html=True)
    st.warning(fact)
    if st.button("Run Hallucination Detection"):
        verdict = detect_hallucination(fact, response)
        st.markdown('<div class="section-header">Hallucination Verdict</div>', unsafe_allow_html=True)
        if "Not Hallucinated" in verdict:
            st.markdown(f'<div class="verdict-box" style="background-color:#D4EDDA;color:#155724;"> {verdict}</div>', unsafe_allow_html=True)
        elif "Hallucinated" in verdict:
            st.markdown(f'<div class="verdict-box" style="background-color:#F8D7DA;color:#721C24;"> {verdict}</div>', unsafe_allow_html=True)
        else:
            st.error(verdict)