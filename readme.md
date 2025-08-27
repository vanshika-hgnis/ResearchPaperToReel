# ResearchPapertoReel

- AI-Powered Academic Explainer
- converts research paper to explainable reel like videos of 5 mins bites (educational content)

# 1) Create env (recommended Python 3.10+)

python -m venv .venv && . .venv/bin/activate # Windows: .venv\Scripts\activate

# 2) Install deps

pip install -U -r requirements.txt

# 3) Add keys (choose any provider you have)

cp .env.example .env # then put your API key(s)

# 4) Run the pipeline (summarize + write Manim scene)

python paper2short.py --pdf ./sample.pdf --max-secs 15 --lang en \
--provider mistral --model mistral-small-latest # or: --provider openai --model gpt-4o-mini

# 5) Render the video with Manim (vertical 1080x1920)

manim -pqh reel_scene.py ReelScene

# Output under: media/videos/reel_scene/1080p60/ReelScene.mp4
