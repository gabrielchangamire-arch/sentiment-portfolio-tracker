"""
Configuration — loads settings from a .env file or environment variables.

When NEWSAPI_KEY is not set the app runs in demo mode with sample data.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads from .env in the project root

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
DEMO_MODE = not bool(NEWSAPI_KEY)
