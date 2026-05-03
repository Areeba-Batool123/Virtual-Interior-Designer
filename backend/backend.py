import os
import base64
import traceback
import json
import re
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# ================= ENV =================
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"  # Groq model (currently available)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= PERSISTENT MEMORY =================
MEMORY_FILE = Path(__file__).parent / "session_memory.json"
IMAGE_FILE  = Path(__file__).parent / "session_image.jpg"
FRONTEND_FILE = Path(__file__).parent.parent / "frontend" / "frontend.html"

def save_memory(image_prompt: str, budget: str):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"image_prompt": image_prompt, "budget": budget}, f)

def load_memory():
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                return data.get("image_prompt"), data.get("budget")
        except:
            pass
    return None, None

def save_image_bytes(data: bytes):
    with open(IMAGE_FILE, "wb") as f:
        f.write(data)

def load_image_bytes() -> Optional[bytes]:
    if IMAGE_FILE.exists():
        with open(IMAGE_FILE, "rb") as f:
            return f.read()
    return None

last_image_prompt, last_budget = load_memory()

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """You are a professional AI interior designer.

RULES:
- Keep room layout SAME
- Do NOT move furniture unless asked
- Apply ONLY requested changes

SMART COMMANDS:
- "luxury" -> premium materials, warm lighting
- "clean/minimal" -> declutter, neutral tones
- "cozy" -> soft lighting, textures

BUDGET RULE:
- Respect user budget strictly
- Suggest affordable items

OUTPUT FORMAT:

🏠 summary
🛋️ furniture
🎨 palette
💰 budget breakdown (if budget given)
📐 tips
✨ vibe

IMAGE_PROMPT: <single line detailed prompt>
"""

# ================= REQUEST =================
class ChatRequest(BaseModel):
    message:   str
    style:     str
    color:     str
    room_type: str
    budget:    Optional[int] = None

# ================= HELPERS =================
def extract_image_prompt(text: str) -> str:
    for line in text.split("\n"):
        clean = re.sub(r'\*+', '', line).strip()
        clean = re.sub(r'^[^\x00-\x7F]+', '', clean).strip()
        if "IMAGE_PROMPT:" in clean:
            after = clean.split("IMAGE_PROMPT:")[1].strip()
            if after:
                return after
    match = re.search(r'IMAGE_PROMPT:\s*(.+)', text)
    if match:
        return match.group(1).strip()
    return ""

def clean_response(text: str) -> str:
    return "\n".join(
        line for line in text.split("\n")
        if "IMAGE_PROMPT:" not in line
    ).strip()

def generate_image_from_prompt(prompt: str) -> Optional[str]:
    """Fallback: text-to-image via Pollinations with retry."""
    if not prompt:
        return None
    # Trim prompt to avoid URL too long errors
    trimmed = prompt[:300]
    full_prompt = trimmed + ", photorealistic, 4k, interior photography, professional lighting"
    
    for attempt in range(3):  # retry up to 3 times
        try:
            print(f"[Pollinations] Attempt {attempt+1}...")
            response = httpx.get(
                f"https://image.pollinations.ai/prompt/{full_prompt}",
                params={"width": 800, "height": 500, "nologo": "true", "seed": 42},
                timeout=90.0,
                follow_redirects=True
            )
            if response.status_code == 200:
                print(f"[Pollinations] Success! {len(response.content)} bytes")
                return base64.b64encode(response.content).decode()
            print(f"[Pollinations] HTTP {response.status_code}")
        except httpx.TimeoutException:
            print(f"[Pollinations] Timeout on attempt {attempt+1}, retrying...")
        except Exception as e:
            print(f"[Pollinations ERROR] {e}")
            break
    return None

def edit_image_with_groq(image_bytes: bytes, edit_prompt: str) -> Optional[str]:
    """
    Groq does not support image generation/editing, so always return None.
    Fallback to Pollinations for image generation.
    """
    return None

def smart_enhance(msg: str) -> str:
    msg   = msg.lower()
    extra = ""
    if "luxury" in msg:
        extra += "luxury materials, premium furniture, warm lighting, "
    if "clean" in msg or "minimal" in msg:
        extra += "minimal design, decluttered space, neutral palette, "
    if "cozy" in msg:
        extra += "soft lighting, cozy textures, warm tones, "
    return extra

def object_enforce(msg: str) -> str:
    msg = msg.lower()
    if "dustbin" in msg or "bin" in msg:
        obj = "white dustbin"
        if "left"  in msg: obj += " on left side"
        if "right" in msg: obj += " on right side"
        if "under" in msg: obj += " under desk"
        return f"{obj}, {obj}, clearly visible {obj}"
    return ""

def parse_budget_value(value: Optional[int | str]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(float(str(value).strip()))
        return parsed if parsed > 0 else None
    except Exception:
        return None

def apply_intent_budget_adjustments(estimated_cost: int, breakdown: dict, user_message: Optional[str]):
    if not user_message:
        return estimated_cost, breakdown, None

    msg = user_message.lower()
    adjusted = dict(breakdown)
    reduction = 0.0
    increase = 0.0
    notes = []

    # Cost-cutting intent
    if any(k in msg for k in ["remove", "rmove", "delete", "without", "less", "cut", "reduce"]):
        reduction += 0.10
        notes.append("Applied cost-cutting request")
    if any(k in msg for k in ["cheap", "budget", "affordable", "low cost", "within budget"]):
        reduction += 0.08
        notes.append("Switched to budget-friendly options")

    # Premium intent
    if any(k in msg for k in ["luxury", "premium", "high end", "expensive"]):
        increase += 0.12
        notes.append("Premium material preference increased cost")

    # Category-specific adjustments from user message
    if any(k in msg for k in ["table", "coffee table"]):
        adjusted["Furniture"] = round(adjusted["Furniture"] * 0.90)
        notes.append("Reduced furniture cost (table removal/change)")
    if any(k in msg for k in ["sofa", "couch"]):
        adjusted["Furniture"] = round(adjusted["Furniture"] * 0.88)
        notes.append("Reduced furniture cost (sofa change)")
    if any(k in msg for k in ["accessories", "decor", "decoration"]):
        adjusted["Decor"] = round(adjusted["Decor"] * 0.75)
        notes.append("Reduced decor/accessories budget")
    if any(k in msg for k in ["lighting", "lights", "lamp"]):
        adjusted["Lighting"] = round(adjusted["Lighting"] * 0.90)
        notes.append("Reduced lighting budget")

    net_factor = 1.0 - min(reduction, 0.30) + min(increase, 0.25)
    adjusted_total = round(sum(adjusted.values()) * net_factor)

    # Keep breakdown aligned to adjusted total using a scaling pass
    current_total = max(1, sum(adjusted.values()))
    scale = adjusted_total / current_total
    for key in list(adjusted.keys()):
        adjusted[key] = round(adjusted[key] * scale)

    note = "; ".join(notes) if notes else None
    return adjusted_total, adjusted, note

def build_budget_insights(budget_value: Optional[int], room_type: str, style: str, user_message: Optional[str] = None):
    if not budget_value:
        return None

    room_factor = {
        "bedroom": 0.95,
        "living room": 1.10,
        "dining room": 1.05,
        "home office": 0.90,
        "kitchen": 1.25,
        "bathroom": 1.15,
    }.get((room_type or "").lower(), 1.0)

    style_factor = {
        "minimal": 0.92,
        "scandinavian": 0.97,
        "japandi": 0.98,
        "modern": 1.00,
        "bohemian": 1.05,
        "industrial": 1.08,
        "traditional": 1.10,
        "luxury": 1.25,
    }.get((style or "").lower(), 1.0)

    estimated_cost = int(round(budget_value * room_factor * style_factor * 1.18 / 1000.0)) * 1000
    breakdown = {
        "Furniture": round(estimated_cost * 0.40),
        "Lighting": round(estimated_cost * 0.15),
        "Decor": round(estimated_cost * 0.15),
        "Paint / Walls": round(estimated_cost * 0.10),
        "Contingency": round(estimated_cost * 0.20),
    }

    estimated_cost, breakdown, note = apply_intent_budget_adjustments(estimated_cost, breakdown, user_message)
    difference = estimated_cost - budget_value

    return {
        "budget": budget_value,
        "estimated_cost": estimated_cost,
        "difference": difference,
        "warning": difference > 0,
        "breakdown": breakdown,
        "note": note,
    }

# ================= ROOT =================
@app.get("/")
def root():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    return {"status": "running 🚀"}

@app.get("/status")
def status():
    img_prompt, budget = load_memory()
    has_image = IMAGE_FILE.exists()
    return {
        "has_room":   bool(img_prompt),
        "has_image":  has_image,
        "budget":     budget,
        "preview":    (img_prompt[:80] + "...") if img_prompt else None
    }

# ================= ANALYZE =================
@app.post("/analyze")
async def analyze(
    image:     UploadFile    = File(...),
    style:     str           = Form(...),
    color:     str           = Form(...),
    room_type: str           = Form(...),
    budget:    Optional[str] = Form(None)
):
    global last_image_prompt, last_budget

    try:
        image_bytes = await image.read()
        budget_value = parse_budget_value(budget)

        # Save original image to disk for later edits
        save_image_bytes(image_bytes)
        print(f"[analyze] Saved original image: {len(image_bytes)} bytes")

        prompt = f"""{SYSTEM_PROMPT}

Room: {room_type}
Style: {style}
Color: {color}
Budget: {budget_value}

Analyze and redesign carefully. Keep layout same.
IMPORTANT: End your response with IMAGE_PROMPT: on its own line.
"""

        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )

        text = res.choices[0].message.content
        image_prompt = extract_image_prompt(text)
        clean_text   = clean_response(text)

        print(f"[analyze] Extracted prompt: '{image_prompt[:80] if image_prompt else 'EMPTY'}'")

        last_image_prompt = image_prompt
        budget_insights = build_budget_insights(budget_value, room_type, style, None)

        last_budget       = str(budget_value) if budget_value else budget
        save_memory(image_prompt, str(budget_value) if budget_value else (budget or ""))

        # For analyze: use Pollinations (full styled redesign)
        img = generate_image_from_prompt(image_prompt)

        return {
            "response":     clean_text,
            "image":        img,
            "image_prompt": image_prompt,
            "budget_insights": budget_insights,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ================= CHAT =================
@app.post("/chat")
async def chat(req: ChatRequest):
    global last_image_prompt, last_budget

    if not last_image_prompt:
        last_image_prompt, last_budget = load_memory()

    try:
        if not last_image_prompt:
            raise HTTPException(
                status_code=400,
                detail="Pehle image upload karo aur Analyze karo"
            )

        smart = smart_enhance(req.message)
        obj   = object_enforce(req.message)
        budget_value = parse_budget_value(req.budget or last_budget)

        extras = ", ".join(filter(None, [smart.strip(", "), obj.strip(", ")]))
        edit_description = req.message + (", " + extras if extras else "")

        context = f"Existing room style: {last_image_prompt}"

        prompt = f"""{SYSTEM_PROMPT}

Room: {req.room_type}
Style: {req.style}
Color: {req.color}
Budget: {budget_value}

{context}

User Request: {req.message}

IMPORTANT: End your response with IMAGE_PROMPT: on its own line.
"""

        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )

        text = res.choices[0].message.content
        image_prompt = extract_image_prompt(text)
        clean_text   = clean_response(text)

        if image_prompt:
            last_image_prompt = image_prompt
            save_memory(last_image_prompt, str(budget_value or last_budget or ""))

        budget_insights = build_budget_insights(budget_value, req.room_type, req.style, req.message)

        # Groq doesn't support image editing, so always use Pollinations
        img = generate_image_from_prompt(last_image_prompt)

        return {
            "response":     clean_text,
            "image":        img,
            "image_prompt": last_image_prompt,
            "budget_insights": budget_insights,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))