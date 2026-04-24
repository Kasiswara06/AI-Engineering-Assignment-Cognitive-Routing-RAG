# AI Cognitive Routing & RAG System

## Overview
This project implements:
- Vector-based persona routing
- LangGraph autonomous content generation
- RAG-based argument defense with injection protection

---

## Phase 1: Persona Matching
- Used FAISS vector DB
- HuggingFace embeddings
- Cosine similarity for routing posts to relevant bots

---

## Phase 2: LangGraph Pipeline
Nodes:
1. Decide Topic (LLM chooses topic)
2. Web Search (mock search tool)
3. Draft Post (LLM generates JSON output)

---

## Phase 3: Combat Engine
- Uses full conversation context (RAG-style prompt)
- Includes system-level guardrails:
  - Ignore prompt injections
  - Maintain persona strictly

---

## Setup Instructions

### Install dependencies
pip install -r requirements.txt

### Install Ollama
Download and run:
https://ollama.com

Then:
ollama pull llama3

---

## Run

Phase 1:
python phase1_router.py

Phase 2:
python phase2_langgraph.py

Phase 3:
python phase3_combat.py