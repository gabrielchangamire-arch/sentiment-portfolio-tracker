"""
Configuration — loads settings from a .env file or environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads from .env in the project root

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

if not NEWSAPI_KEY:
    print("WARNING: NEWSAPI_KEY is not set. Create a .env file with:\n"
          '  NEWSAPI_KEY=your_key_here\n'
          "or export it as an environment variable.")
