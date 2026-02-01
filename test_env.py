from dotenv import load_dotenv
import os
load_dotenv()
print("GEMINI_API_KEY loaded?", os.getenv("GEMINI_API_KEY") is not None)
