"""
Step 2 of the scraping pipeline: VERIFY each raw candidate with Gemini vision.

This is the only step that needs intelligence — a search for "warping" returns
plenty of wrong-colour, wrong-defect, macro-lab, render, and meme images. We show
each candidate to Gemini with the class definition + image spec and get back a
structured judgement (class + filament colour + view + a gold flag + confidence).
run.py then applies the deterministic gating (target-class match, gold rejection,
per-colour diversity cap) on top of this assessment.

Uses Google's google-genai SDK. Default model gemini-2.5-flash (free API tier via
Google AI Studio — get a key at aistudio.google.com). Set GEMINI_API_KEY (or
GOOGLE_API_KEY) in your environment. Override the model with SCRAPER_VERIFY_MODEL
(e.g. gemini-2.5-pro for maximum accuracy on borderline classes).

Install: pip install google-genai
"""

import json
import os
import re
import time
from enum import Enum

from pydantic import BaseModel
from google import genai
from google.genai import types

from config import IMAGE_SPEC, VERIFY_MODEL

# Client reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
_client = genai.Client(api_key=_api_key) if _api_key else genai.Client()


# --- Structured-output schema (Pydantic -> Gemini response_schema) ----------
class _DefectClass(str, Enum):
    cracking = "cracking"
    warping = "warping"
    off_platform = "off_platform"
    blob_of_death = "blob_of_death"
    spaghetti = "spaghetti"
    stringing = "stringing"
    layer_shifting = "layer_shifting"
    none = "none"


class _View(str, Enum):
    webcam_above_bed = "webcam_above_bed"
    wide_angle = "wide_angle"
    angled = "angled"
    macro_closeup = "macro_closeup"
    other = "other"


class _VerifyResult(BaseModel):
    shows_3d_print: bool
    defect_class: _DefectClass
    filament_colour: str
    is_gold_metallic: bool
    view: _View
    meets_image_spec: bool
    is_render_or_marketing: bool
    confidence: float
    reason: str


_SYSTEM = (
    "You are a meticulous dataset curator for a 3D-printing defect-detection model. "
    "You judge whether a single image is a good training example for a specific target. "
    "Be strict: it is far better to reject a borderline image than to admit a mislabelled "
    "one — bad labels are what previously biased this model. Report the filament colour you "
    "actually see, and set is_gold_metallic=true for any gold/brass/bronze/copper/metallic part."
)


def verify_image(image_path, spec: dict) -> dict:
    """Return Gemini's structured assessment of one image as a plain dict.

    Raises on hard API failures; callers (run.py) handle/skip.
    """
    view_pref = spec.get("view_pref", "")
    prompt = (
        f"TARGET DEFINITION:\n{spec['definition']}\n\n"
        f"{IMAGE_SPEC}\n\n"
        + (f"PREFERRED VIEW FOR THIS DEFECT: {view_pref}\n\n" if view_pref else "")
        + "Assess the attached image against the target definition and the image "
        "requirements, and fill in the schema. `defect_class` is the SINGLE best-matching "
        "defect you actually see (or 'none' if the print is clean / there is no print). If the "
        "image could plausibly fit more than one class, or the defect is not the clear, dominant "
        "subject, lower your confidence. When unsure whether it truly matches the TARGET "
        "definition, prefer a low confidence or 'none' — rejecting a good image costs us nothing, "
        "but a wrong label costs us a lot. `confidence` is 0.0-1.0."
    )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Free-tier quotas are per-minute (e.g. flash-lite: 10 RPM). A 429 means
    # "wait", not "this image is bad" — so honour the API's retryDelay instead
    # of skipping the candidate. Daily-quota exhaustion still raises after the
    # retries are spent, and run.py skips as before.
    last_err = None
    for attempt in range(6):
        try:
            resp = _client.models.generate_content(
                model=VERIFY_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    response_mime_type="application/json",
                    response_schema=_VerifyResult,
                    temperature=0.0,
                ),
            )
            break
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                raise
            last_err = e
            m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)", msg)
            wait = int(m.group(1)) + 2 if m else 35
            time.sleep(min(wait, 90))
    else:
        raise last_err

    data = json.loads(resp.text)
    try:
        data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    except (TypeError, ValueError):
        data["confidence"] = 0.0
    return data
