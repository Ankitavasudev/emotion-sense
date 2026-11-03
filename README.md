<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B0F14,50:1B4B5A,100:00ADB5&height=170&section=header&text=Emotion%20Sense&fontSize=46&fontColor=EAF1F8&fontAlignY=40&desc=Explainable%20emotion%20scoring%20API&descAlignY=60&descSize=15" width="100%" alt="Emotion Sense" />

**Every prediction lists the exact words that produced it.**

<br />

<a href="https://ankitavasudev.github.io/emotion-sense/"><img src="https://img.shields.io/badge/%E2%96%B6%20TRY%20IT-00ADB5?style=for-the-badge&logoColor=white" alt="Live demo" /></a>
&nbsp;
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB" />
<img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/tests-25%20passing-3FB950?style=flat-square" />

</div>

---

## Why this exists

Most sentiment demos wrap a model you cannot inspect: you send text, you get a label, and there is no way to ask *why*. This one is built the other way around. It is a transparent scoring engine where every number traces back to tokens in your sentence, wrapped in a real FastAPI service with a React dashboard.

```console
$ curl -s -X POST localhost:8000/api/v1/analyze \
    -H 'content-type: application/json' \
    -d '{"text":"I am not happy with this result"}' | jq '.data | {emotion, confidence, signals}'

{
  "emotion": "sad",
  "confidence": 0.498,
  "signals": [
    { "token": "happy", "emotion": "sad", "source_emotion": "happy", "weight": 0.96, "negated": true }
  ]
}
```

Note what happened: `happy` was detected, `not` inverted it, and the response says so. That is the whole point.

## Try it without installing anything

The [browser demo](https://ankitavasudev.github.io/emotion-sense/) runs
`backend/ml/model.py` unmodified through [Pyodide](https://pyodide.org/), so the
page scores text with the same Python code as the API. Nothing is uploaded. The
first load downloads the Python runtime (a few MB) and is cached afterwards.

## Honest accuracy

The engine is a lexical classifier, not a transformer, and the README says so:

| Metric | Value |
| :-- | :-- |
| Accuracy on the bundled sanity set | **93.1%** (67 / 72 sentences) |
| Classes | happy · sad · angry · fear · surprise · neutral |
| Latency | single-digit milliseconds, no GPU |
| Network calls at inference | none |

The 72-sentence set lives in `backend/eval/emotion_eval.py`, so the number is reproducible:

```bash
cd backend && python -m eval.run_eval
```

It is a hand-written sanity set, not a benchmark. Expect lower scores on sarcasm, third-party news text, and anything where emotion is implied rather than stated. The known failure modes are listed at the bottom instead of being hidden.

## What the engine actually does

1. **Tokenise** with word boundaries — the old substring matcher fired `mad` on `made` and `miss` on `missing`. Both are now regression tests.
2. **Score** each token against a weighted lexicon (six emotions, ~140 terms, strong words like *furious* outweigh mild ones like *okay*).
3. **Scale by intensity** — `extremely good` outranks `good`, `slightly annoyed` underranks `annoyed`.
4. **Invert on negation** within a clause — `not happy` becomes sad, but the scan stops at `but`, `just` and other clause boundaries, so `not angry, just disappointed` correctly reads as disappointed.
5. **Calibrate** the raw weights with a softmax, so `confidence` is a probability distribution rather than `hits / total_hits`.

An optional scikit-learn pipeline at `backend/models/sentiment_model.pkl` takes over when present; the lexicon engine is the fallback and the default.

## Stack

| Layer | Choice |
| :-- | :-- |
| API | FastAPI + Pydantic v2 |
| Engine | Pure Python (stdlib only) |
| Dashboard | React + Chart.js + Framer Motion, with the signal list shown under every result |
| Packaging | Docker + docker-compose |
| CI | GitHub Actions: `pytest`, the frontend component tests and a production build |
| Demo | GitHub Pages, running the Python engine in the browser via Pyodide |

## Quick start

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload        # http://localhost:8000/docs

# frontend
cd frontend
npm install && npm start         # http://localhost:3000
```

Or the whole thing:

```bash
docker compose up --build
```

## API

| Method | Endpoint | Purpose |
| :-- | :-- | :-- |
| `GET` | `/` | API metadata |
| `GET` | `/health` | Liveness + which engine is active |
| `POST` | `/api/v1/analyze` | One text → emotion, confidence, scores, **signals** |
| `POST` | `/api/v1/analyze/batch` | Up to 100 texts + distribution summary |
| `GET` | `/api/v1/history` | Recent analyses, paginated |

Interactive docs are generated at `/docs`.

## Tests

```bash
cd backend
pytest -q                         # 19 backend tests

cd ../frontend
npm test -- --watchAll=false      # 6 component tests
```

The backend suite covers the bugs this rewrite was written to kill: substring
false positives, negation, clause-scoped negation, intensifier ordering,
probability normalisation, explainability, and an accuracy floor so the engine
cannot silently regress. The component tests check that the signal list is
rendered and that clicking an example analyses that example rather than the
previous contents of the box.

## Known limitations

- **Sarcasm** — `"Great, another outage"` scores as happy.
- **Implicit affect** — `"No issues at all, smooth release"` carries no lexicon hit, so it lands neutral.
- **No context** — each text is scored alone; a reply cannot change the reading of the message before it.
- **English only** — the lexicon is English.

These are deliberate trade-offs for auditability, not oversights.

---

<div align="center">

**Ankita Salaria** · [@Ankitavasudev](https://github.com/Ankitavasudev)

</div>
