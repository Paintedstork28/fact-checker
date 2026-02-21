"""Configuration and constants for the Fact Checker."""
import os

# Gemini API key: prefer Streamlit secrets, then env var, then local config
def get_gemini_api_key():
    try:
        import streamlit as st
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    import json
    settings_path = os.path.expanduser("~/.gemini/settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            return json.load(f).get("GEMINI_API_KEY", "")
    return ""

GEMINI_MODEL = "gemini-2.5-flash"
SEARCH_RESULTS_PER_QUERY = 5
SOURCE_SCORE_THRESHOLD = 5
MAX_RESEARCHER_RETRIES = 2
