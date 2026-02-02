import os, json, mimetypes, sys
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from google.genai import types
from pathlib import Path

BASE_DIR = Path(__file__).parent
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

#Gemini 2.5 flash is used for planning and reasoning
client = genai.Client(api_key=API_KEY)
MODEL = os.getenv("GEMINI_MODEL")
# default to gemini-2.5-flash if not set in .env
if not MODEL:
    raise RuntimeError('Missing GEMINI_MODEL in .env (e.g. "gemini-2.5-flash" or "gemini-2.5-pro")')



# config for the model 
GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.3,
    top_p=0.95,
    max_output_tokens=512,
    response_mime_type="application/json",
)


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

    # if case has image_paths, add the images to the prompt
    image_list = []
    if case.get("image_paths"):
      image_list = case["image_paths"]
    elif case.get("image_path"):
      image_list = [case["image_path"]]

    # if no images, generate a response from the prompt
    if not image_list:
        response = client.models.generate_content(model=MODEL, contents=[full_prompt], config=GEN_CONFIG)
        print(response.text)
        return

    # if there are images, add them to the prompt (paths relative to project root)
    contents = [full_prompt] + [
        image_part_from_path(str(BASE_DIR / img) if not os.path.isabs(img) else img)
        for img in image_list
    ]
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=GEN_CONFIG
    )
    print(response.text)

  

if __name__ == "__main__":
    case_path = sys.argv[1] if len(sys.argv) > 1 else "cases/case1.json"
    run(case_path)
