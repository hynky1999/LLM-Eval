import os
from dotenv import load_dotenv

load_dotenv()


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
