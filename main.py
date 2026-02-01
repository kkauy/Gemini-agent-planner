import os, json, mimetypes, sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path

load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

#Gemini 3 pro is used for planning and reasoning
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3-pro"

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_case(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def image_part_from_path(image_path: str):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return types.Part.from_bytes(data=img_bytes, mime_type=mime_type)  
def run(case_path: str):
    case = read_case(case_path)
    BASE_DIR = Path(__file__).parent
    prompt = read_text(BASE_DIR / "prompt.txt")


    # add the structured case to the prompt
    full_prompt = prompt + "\n\nINPUT_JSON:\n" + json.dumps(case, ensure_ascii=False)

    contents = [full_prompt]

    # if case has image_path, add the image to the prompt
    if "image_path" in case and case["image_path"]:
        contents.append(image_part_from_path(case["image_path"]))

    response = client.models.generate_content(
        model=MODEL,
        contents=contents
    )

    print(response.text)

if __name__ == "__main__":
    case_path = sys.argv[1] if len(sys.argv) > 1 else "cases/case1.json"
    run(case_path)
