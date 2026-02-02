import os, json, mimetypes
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError



# --------------------
# UI
# --------------------
st.set_page_config(page_title="PetSense", page_icon="🐾", layout="centered")
st.title("🐾PetSense — Emotion & Behavior Interpreter")
st.caption("Upload a pet photo + behavioral signals → get a single emotion label + actions (STRICT JSON).")

# --------------------
# Setup
# --------------------
BASE_DIR = Path(__file__).parent
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")
if not MODEL:
    st.error("Missing GEMINI_MODEL in .env (e.g. gemini-2.5-pro)")
    st.stop()

if not API_KEY:
    st.error("Missing GEMINI_API_KEY in .env")
    st.stop()

client = genai.Client(api_key=API_KEY)

GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.3,
    top_p=0.95,
    max_output_tokens=512,
    response_mime_type="application/json",
)


LABELS = ["excitement","alert","anxiety","fear","boredom","discomfort","neutral","unknown"]

# required keys for the output
REQUIRED_KEYS = [
    "pet_profile",
    "observations",
    "interpreted_state",
    "confidence",
    "reasoning",
    "recommended_action",
    "fallback_action",
]


SAFE_FALLBACK = {
    "pet_profile": {"species": "dog", "age": "", "weight": ""},
    "observations": [],
    "interpreted_state": "unknown",
    "confidence": 0.2,
    "reasoning": ["Insufficient reliable structured output; try clearer inputs."],
    "recommended_action": "Provide clearer signals/photo and retry.",
    "fallback_action": "Reduce stimulation and observe for 3–5 minutes.",
}


def validate_output(obj: dict):
    """Return (ok: bool, err: str)."""
    if not isinstance(obj, dict):
        return False, "Output is not a JSON object"

    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        return False, f"Missing keys: {missing}"

    if not isinstance(obj.get("pet_profile"), dict):
        return False, "pet_profile must be an object"
    if not isinstance(obj.get("observations"), list):
        return False, "observations must be an array"

    state = obj.get("interpreted_state")
    if state not in LABELS:
        return False, f"Invalid interpreted_state: {state}"

    conf = obj.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        return False, "confidence is not a number"
    if not (0.0 <= conf <= 1.0):
        return False, "confidence out of range 0-1"

    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, list) or not all(isinstance(x, str) for x in reasoning):
        return False, "reasoning must be an array of strings"

    if not isinstance(obj.get("recommended_action"), str):
        return False, "recommended_action must be a string"

    # Also add validation for fallback_action since it's in REQUIRED_KEYS
    if not isinstance(obj.get("fallback_action"), str):
        return False, "fallback_action must be a string"
        
    if not isinstance(obj.get("pet_profile"), dict):
        return False, "pet_profile must be an object"

    if not isinstance(obj.get("observations"), list):
        return False, "observations must be an array"
    if len(reasoning) > 3:
        return False, "reasoning must have at most 3 items"

    return True, None    

MAX_RETRIES = 2  

def run_agent_with_retry(prompt: str, infer_fn):
    last_err = None
    last_text = None

    for attempt in range(MAX_RETRIES + 1):
        text, err = infer_fn(prompt)
        last_text = text

        if err:
            last_err = f"Gemini error: {err}"
            if "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                break
            continue

        obj, parse_err = safe_parse_json(text)
        if parse_err:
            last_err = parse_err
        else:
            ok, v_err = validate_output(obj)
            if ok:
                return obj, {"status": "ok", "attempt": attempt, "error": None, "raw_text": text}
            last_err = v_err

        prompt = (
            prompt
            + "\n\nYour previous output was invalid because: "
            + str(last_err)
            + "\nReturn ONLY valid JSON. Include ALL required keys exactly: "
            + ", ".join(REQUIRED_KEYS)
            + ". reasoning must be a JSON array of strings (no 0:). No extra text."
        )

    return SAFE_FALLBACK, {"status": "fallback", "attempt": MAX_RETRIES + 1, "error": last_err, "raw_text": last_text}


# helper function to normalize the reasoning output
def normalize_reasoning(x):
    # Want: list[str]
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, dict):
        # sort keys so 0,1,2 order stable
        try:
            items = [x[k] for k in sorted(x.keys(), key=lambda z: int(z) if str(z).isdigit() else str(z))]
        except Exception:
            items = list(x.values())
        return [str(i) for i in items]
    return [str(x)]

def read_prompt() -> str:
    return (BASE_DIR / "prompt.txt").read_text(encoding="utf-8")

def image_part_from_bytes(img_bytes: bytes, filename: str):
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = "image/jpeg"
    return types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

def extract_json_from_response(text: str) -> str:
    """Get JSON string from model output (JSON may be wrapped in markdown or have leading text)."""
    if not text or not text.strip():
        return ""
    text = text.strip()

    # Strip markdown code block if present
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()
    if text.startswith("```"):
        start = text.index("\n", 3) + 1 if "\n" in text[3:] else 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        return text[start:end].strip()

    # Find first { and try balanced braces to end of JSON object
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    quote = None
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if c == quote:
                in_string = False
            continue
        if c in ('"', "'"):
            in_string = True
            quote = c
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]

def safe_parse_json(text: str):
    """Try parse JSON; return (obj, err)."""
    if not text or not text.strip():
        return None, "Empty response"
    extracted = extract_json_from_response(text)
    try:
        return json.loads(extracted), None
    except json.JSONDecodeError as e:
        repaired = _try_repair_truncated_json(extracted)
        if repaired:
            try:
                return json.loads(repaired), None
            except json.JSONDecodeError:
                pass
        return None, str(e)

def _try_repair_truncated_json(s: str) -> Optional[str]:
    """Attempt to close truncated JSON. Returns only if repaired string parses as valid JSON."""
    s = s.strip()
    if not s.startswith("{"):
        return None
    stack = []  # '[' or '{'
    in_string = False
    escape = False
    i = 0
    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == '\\':
            escape = True
            i += 1
            continue
        if in_string:
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == '{':
            stack.append('}')
            i += 1
            continue
        if c == '[':
            stack.append(']')
            i += 1
            continue
        if c == '}' or c == ']':
            if stack and stack[-1] == c:
                stack.pop()
            i += 1
            continue
        i += 1
    if in_string:
        s += '"'
    s += "".join(reversed(stack))
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        return None

def call_gemini(prompt: str, case_obj: dict, image_bytes_list: list):
    full_prompt = prompt + "\n\nINPUT_JSON:\n" + json.dumps(case_obj, ensure_ascii=False)

    contents = [full_prompt]
    for (b, fname) in image_bytes_list:
        contents.append(image_part_from_bytes(b, fname))

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=GEN_CONFIG
        )
        return resp.text, None
    except ClientError as e:
        return None, str(e)



with st.expander(" Model Settings", expanded=False):
    st.write(f"Model: `{MODEL}`")
    st.write(f"temperature: `{GEN_CONFIG.temperature}` | top_p: `{GEN_CONFIG.top_p}` | max_output_tokens: `{GEN_CONFIG.max_output_tokens}`")

prompt = read_prompt()

st.subheader("1) Pet profile (optional)")
col1, col2, col3 = st.columns(3)
species = col1.text_input("species", value="dog")
age = col2.text_input("age", value="")
weight = col3.text_input("weight", value="")

st.subheader("2) Behavioral signals (structured)")
st.caption("Keep it observable. Avoid medical claims.")

signals = {}
c1, c2, c3 = st.columns(3)
signals["tail_wagging"] = c1.selectbox("tail_wagging", ["unknown","none","low","medium","high"], index=0)
signals["ears_position"] = c2.selectbox("ears_position", ["unknown","forward","neutral","back"], index=0)
signals["body_posture"] = c3.selectbox("body_posture", ["unknown","relaxed","tense","crouched","stiff"], index=0)

c4, c5, c6 = st.columns(3)
signals["vocalization"] = c4.selectbox("vocalization", ["unknown","none","barking","whining","growling","meowing"], index=0)
signals["pacing"] = c5.selectbox("pacing", ["unknown","no","yes"], index=0)
signals["hiding"] = c6.selectbox("hiding", ["unknown","no","yes"], index=0)

notes = st.text_area("free_notes (optional)", value="", height=80)

st.subheader("3) Upload image(s) (optional)")
uploads = st.file_uploader("Upload 1–3 images", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)

image_bytes_list = []
if uploads:
    for up in uploads[:3]:
        b = up.read()
        image_bytes_list.append((b, up.name))
    st.image([b for (b, _) in image_bytes_list], caption=[n for (_, n) in image_bytes_list], width=220)

case_obj = {
    "pet_profile": {"species": species, "age": age, "weight": weight},
    "behavioral_signals": signals,
    "free_notes": notes
}

st.subheader("4) Run inference")
if "next_allowed" not in st.session_state:
    st.session_state.next_allowed = 0.0

cooldown_active = time.time() < st.session_state.next_allowed
run = st.button("Analyze emotion", type="primary", disabled=cooldown_active)

if cooldown_active:
    wait_s = int(st.session_state.next_allowed - time.time())
    st.info(f"Cooling down to avoid quota. Try again in {wait_s}s.")

if run:
    with st.spinner("Calling Gemini..."):
        infer_fn = lambda p: call_gemini(p, case_obj, image_bytes_list)
        obj, meta = run_agent_with_retry(prompt, infer_fn)

    used_fallback = (meta["status"] == "fallback")

    if used_fallback:
        st.warning("Output not valid JSON. Using safe fallback for demo.")
        st.caption(f"Reason: {meta['error']}")
        st.code(meta.get("raw_text") or "", language="text")

        # ONLY rebuild obj when fallback
        obj = {
            "pet_profile": case_obj.get("pet_profile", {"species": "", "age": "", "weight": ""}),
            "observations": [],
            "interpreted_state": "unknown",
            "confidence": 0.2,
            "reasoning": ["Insufficient reliable structured output; try clearer inputs."],
            "recommended_action": "Provide clearer signals/photo and retry.",
            "fallback_action": "Reduce stimulation and observe for 3–5 minutes.",
        }
    else:
        st.success("Parsed JSON successfully")

    # ---- DEBUG ----
    st.markdown("### DEBUG: obj type + keys")
    st.write(type(obj))
    st.write(list(obj.keys()))

    # interpreted_state
    state = obj.get("interpreted_state", "unknown")
    conf = obj.get("confidence", 0.0)

    if state not in LABELS:
        state = "unknown"
    try:
        conf = float(conf)
        conf = max(0.0, min(1.0, conf))
    except:
        conf = 0.0

    st.markdown("### Result")
    st.write(f"**interpreted_state:** `{state}`")

    st.progress(conf)
    st.write(f"**confidence:** `{conf:.2f}`")

    st.markdown("### Recommended actions")
    st.write("**Primary:**", obj.get("recommended_action", ""))
    st.write("**Fallback:**", obj.get("fallback_action", ""))

    st.markdown("### Reasoning (short)")
    raw_reasoning = obj.get("reasoning")
    st.caption(f"reasoning type: {type(raw_reasoning).__name__}")

    reasoning = normalize_reasoning(raw_reasoning)
    if not reasoning:
        st.write("- (none)")
    else:
        for r in reasoning[:6]:
            st.write("-", r)

    st.markdown("### DEBUG: obj right before st.json")
    st.write(type(obj))
    st.write(obj.get("reasoning"))

    st.markdown("### Full JSON")
    st.json(obj)
