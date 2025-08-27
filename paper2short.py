import os, re, json, argparse, textwrap, math, subprocess
import fitz  # PyMuPDF
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional
from dotenv import load_dotenv


def extract_text(pdf_path: str, max_pages: int = 6) -> str:
    doc = fitz.open(pdf_path)
    pages = min(max_pages, len(doc))
    chunks = []
    for i in range(pages):
        t = doc[i].get_text("text")
        chunks.append(t)
    raw = "\n".join(chunks)
    # Basic cleanup
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"Abstract", "\nAbstract\n", raw, flags=re.I)
    raw = re.sub(r"Introduction", "\nIntroduction\n", raw, flags=re.I)
    return raw.strip()


SCRIPT_PROMPT = (
    "You are a ruthless editor for short-form educational videos. "
    "Given a research paper excerpt, write a script for a 10–20 second vertical reel. "
    "Output strict JSON with keys: hook, bullets (array of 3), closing. "
    "Constraints: no bullet over 12 words; total words ≤ 55; plain language; no citations; no equations. "
)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


def call_mistral(model: str, prompt: str, content: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SCRIPT_PROMPT},
            {"role": "user", "content": f"Paper text:\n\n{content}\n\nReturn JSON."},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(MISTRAL_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def cheap_local_fallback(text: str) -> Dict:
    # Grab the first 6–8 sentences, squeeze to headliners
    sents = re.split(r"(?<=[.!?])\s+", text)[:8]
    base = " ".join(sents)
    # crude keyword pick
    base = re.sub(r"\(.*?\)", "", base)
    words = base.split()
    # take ~55 words max
    words = words[:55]
    small = " ".join(words)
    # split into tiny bullets (~10–12 words)
    w = small.split()

    def chunk(ws, n):
        for i in range(0, len(ws), n):
            yield " ".join(ws[i : i + n])

    bullets = list(chunk(w, max(1, min(12, max(8, len(w) // 4)))))[:3]
    return {
        "hook": (bullets[0] if bullets else "Here’s the core idea, fast."),
        "bullets": bullets[:3]
        or ["Key insight one", "Key result two", "Why it matters"],
        "closing": "That’s the gist—learn more in the paper.",
    }


def summarize_to_script(text: str, provider: str, model: str) -> Dict:
    load_dotenv()
    try:
        out = call_mistral(model, SCRIPT_PROMPT, text, os.environ["MISTRAL_API_KEY"])
        return json.loads(out)
    except Exception as e:
        print("[WARN] Mistral failed using fallback", e)
    return cheap_local_fallback(text)


# -----------------------------
# Timing planner (fit to target seconds)
# -----------------------------


def allocate_durations(script: Dict, max_secs: int = 15) -> List[Dict]:
    hook = script.get("hook", "")
    bullets = script.get("bullets", [])
    closing = script.get("closing", "")

    def secs_for(words):
        return max(1.5, len(words.split()) / 3.5)

    parts = [
        {"role": "hook", "text": hook, "t": secs_for(hook) + 0.5},
    ]
    for b in bullets:
        parts.append({"role": "bullet", "text": b, "t": secs_for(b) + 0.3})
    parts.append({"role": "closing", "text": closing, "t": secs_for(closing) + 0.4})
    total = sum(p["t"] for p in parts)
    if total > max_secs:
        scale = max_secs / total
        for p in parts:
            p["t"] = max(1.2, p["t"] * scale)
    return parts


# -----------------------------
# Write Manim scene file
# -----------------------------

SCENE_TEMPLATE = """
from manim import *

config.pixel_width = 1080
config.pixel_height = 1920
config.frame_rate = 60

BG_DARK = "#0b1220"
BG_ACCENT = "#13233f"
ACCENT = "#00d8ff"
TXT = "#f7f9fc"

segments = {segments_json}

class ReelScene(Scene):
    def construct(self):
        # Background gradient
        bg1 = Rectangle(width=10, height=18).set_fill(BG_DARK, 1).set_stroke(width=0)
        bg2 = Rectangle(width=10, height=18).set_fill(BG_ACCENT, 1).set_stroke(width=0)
        bg2.shift(UP * 3 + RIGHT * 2).scale(1.2).set_opacity(0.65)
        self.add(bg1, bg2)

        # Floating accent circle for subtle motion
        dot = Dot(radius=0.25, color=ACCENT).set_opacity(0.8).shift(UP * 5 + LEFT * 2)
        self.add(dot)
        self.play(dot.animate.shift(DOWN * 0.6), run_time=1.2)

        last = None
        for idx, part in enumerate(segments):
            text = part['text']
            role = part['role']
            dur = float(part['t'])

            # Card behind text
            card = RoundedRectangle(
                corner_radius=0.4,
                width=6.8,
                height=4.2,
                fill_color=BG_DARK,
                fill_opacity=0.4
            )
            if role == 'hook':
                card.set_fill(BG_DARK, 0.55)
            if role == 'closing':
                card.set_fill(BG_DARK, 0.35)

            # Main caption
            fs = 62 if role == 'hook' else 56 if role == 'closing' else 58
            caption = Text(text, font_size=fs, color=TXT, line_spacing=0.7)
            grp = VGroup(card, caption).arrange(aligned_edge=ORIGIN).move_to(ORIGIN)

            if last:
                self.play(FadeOut(last, shift=UP * 0.5), run_time=0.25)
            self.play(FadeIn(grp, shift=DOWN * 0.4), run_time=0.35)

            # Slight parallax drift
            self.play(
                AnimationGroup(
                    grp.animate.shift(UP * 0.10),
                    dot.animate.shift(RIGHT * 0.25 + DOWN * 0.1),
                ),
                run_time=dur
            )

            last = grp
"""


def write_manim_scene(segments: List[Dict], out_path: str = "reel/reel_scene.py"):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            SCENE_TEMPLATE.replace(
                "{segments_json}", json.dumps(segments, ensure_ascii=False)
            )
        )
    print(f"[ok] Wrote Manim scene to {out_path}")


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser(description="PDF → LLM → Manim short")
    ap.add_argument("--pdf", required=True, help="Path to PDF/research paper")
    ap.add_argument(
        "--provider",
        default="mistral",
        choices=["mistral", "openai", "none"],
        help="LLM provider",
    )
    ap.add_argument("--model", default="mistral-small-latest", help="Model name")
    ap.add_argument(
        "--max-secs", type=int, default=15, help="Target duration in seconds"
    )
    ap.add_argument("--lang", default="en", help="Language hint (en/hi/etc)")
    args = ap.parse_args()

    text = extract_text(args.pdf)
    print(f"[info] Extracted {len(text)} chars from {args.pdf}")

    provider = args.provider if args.provider != "none" else ""
    script = summarize_to_script(text, provider, args.model)

    # Language tweak: prepend a short Hindi/English cue if needed
    if args.lang.lower().startswith("hi"):
        script["hook"] = "जल्दी समझें: " + script.get("hook", "")
        script["closing"] = (script.get("closing", "") + " और जानने के लिए पेपर देखें")[:80]

    segments = allocate_durations(script, max_secs=args.max_secs)

    with open("segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print("[ok] Saved segments.json")

    write_manim_scene(segments, out_path="reel_scene.py")

    print("\nNext: render with Manim →")
    print(" manim -pqh reel_scene.py ReelScene\n")


if __name__ == "__main__":
    main()
