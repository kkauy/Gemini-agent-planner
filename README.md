🐾 PetSense — Emotion & Behavior Interpreter

PetSense is a Gemini-powered AI agent that interprets pet emotions from images and structured behavioral signals, returning strict JSON output with safe fallback behavior.

🚀 Quick Start (Local Setup)
1️. Prerequisites

Python 3.10+

pip

A Google Gemini API key
(from Google AI Studio)

2️.Clone the repository
git clone https://github.com/kkauy/Gemini-agent-planner.git
cd Gemini-agent-planner

3️.Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
 .venv\Scripts\activate     Windows

4️.Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt yet, minimum required:

pip install streamlit python-dotenv google-genai pillow

5️.Configure environment variables 
Create a local-only env file:
cp .env.example .env.local


Edit .env.local and add your own API key:

GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.5-pro


**Important**

.env.local is not committed to GitHub

Each judge uses their own API key

API keys can be rotated safely without code changes

6️.Run the app
streamlit run app.py


The app will be available at:

http://localhost:8501


**Approach & Design**

PetSense is designed as a robust, demo-safe AI agent, not just a simple API call wrapper.

The core approach focuses on three principles:

** 1. Structured Inputs → Deterministic Outputs**

Instead of sending free-form prompts only, PetSense combines:

Structured behavioral signals (dropdowns)

Optional multimodal inputs (pet images)

A strict JSON schema required from the model

This reduces ambiguity and improves consistency across runs.

**2️. Strict JSON Contract Enforcement**

The Gemini model is required to return ONLY valid JSON with the following guarantees:

All required keys are present

Types are validated (confidence ∈ [0,1], reasoning = list of strings)

Output labels are constrained to a fixed enum

If the model violates the contract:

The response is rejected

A retry is attempted with corrective feedback

The system falls back safely if needed

This mirrors real-world production LLM safety patterns.

3️. **Safe Fallback & Failure-Tolerant Design**

PetSense is built to never crash during demos.

The agent explicitly handles:

Failure Type	Behavior
Invalid / partial JSON	Retry → fallback
Rate limit (429)	Cooldown + fallback
API key revoked / leaked (403)	Immediate fallback
Empty response	Deterministic fallback

Fallback output is:

Predictable

Schema-valid

Clearly labeled as a fallback in the UI


This guarantees a smooth experience for hackathon judging and live demos.

**End-to-End Flow**

User provides pet profile + behavioral signals

Optional images are attached as multimodal input

Gemini is called with a strict JSON-only prompt

Output is validated against a schema

If invalid → retry with corrective prompt

If still invalid → return safe fallback

UI renders results consistently

## Background

This project was originally built during the Google Gemini Hackathon
and later extended with production-style validation, retries,
and deterministic fallback behavior.
