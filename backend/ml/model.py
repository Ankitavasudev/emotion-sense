"""
Emotion scoring engine.

A transparent, dependency-free affect classifier. It is deliberately not a
neural model: every prediction can be traced back to the exact words that
produced it, which makes it fast, auditable and safe to run anywhere.
If a scikit-learn pipeline is present at ``models/sentiment_model.pkl`` it is
used instead, and the lexicon engine becomes the fallback.

Design notes
------------
* Matching is token based, not substring based. ``"mad"`` no longer fires on
  ``"made"`` and ``"miss"`` no longer fires on ``"missing"``.
* Negation is detected within a three-token window and inverts the emotion.
* Intensity modifiers scale a hit: "extremely happy" outweighs "happy".
* Raw hits are converted to a probability distribution with a softmax, so
  ``confidence`` is a real probability rather than ``hits / total_hits``.
"""

import math
import os
import pickle
import re
from collections import Counter
from typing import Dict, List, Optional

EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐",
}

#: term -> weight, grouped by the emotion it supports.
LEXICON: Dict[str, Dict[str, float]] = {
    "happy": {
        "love": 1.3, "loved": 1.3, "lovely": 1.1, "happy": 1.2, "glad": 1.0,
        "great": 1.0, "amazing": 1.2, "wonderful": 1.3, "excellent": 1.3,
        "fantastic": 1.3, "joy": 1.2, "joyful": 1.2, "beautiful": 1.0,
        "awesome": 1.2, "perfect": 1.3, "best": 1.1, "thank": 0.9,
        "thanks": 0.9, "grateful": 1.2, "blessed": 1.1, "excited": 1.2,
        "delighted": 1.3, "thrilled": 1.3, "ecstatic": 1.4, "blissful": 1.3,
        "enjoy": 1.0, "enjoyed": 1.0, "nice": 0.8, "good": 0.8, "fun": 0.9,
        "proud": 1.1, "hopeful": 1.0, "relieved": 1.1, "smile": 0.9,
        "gladly": 1.0, "peaceful": 1.0, "calm": 0.8, "satisfied": 1.1,
    },
    "sad": {
        "sad": 1.2, "sorry": 0.9, "unfortunate": 1.0, "depressed": 1.4,
        "miserable": 1.4, "heartbroken": 1.5, "disappointed": 1.2,
        "disappointing": 1.2, "regret": 1.1, "lonely": 1.2, "crying": 1.2,
        "tears": 1.0, "grief": 1.4, "grieving": 1.4, "sorrow": 1.3,
        "melancholy": 1.3, "gloomy": 1.1, "hopeless": 1.4, "devastated": 1.5,
        "upset": 1.1, "hurt": 1.0, "unhappy": 1.2, "gloom": 1.0,
        "sadly": 1.1, "worthless": 1.3, "empty": 1.0,
        "cry": 1.2, "cried": 1.2, "cries": 1.2, "poor": 0.7, "sadder": 1.2,
        "unmotivated": 1.2, "discouraged": 1.2, "dejected": 1.3,
    },
    "angry": {
        "angry": 1.3, "terrible": 1.1, "horrible": 1.2, "hate": 1.3,
        "hated": 1.2, "furious": 1.5, "rage": 1.4, "raging": 1.4,
        "annoyed": 1.1, "annoying": 1.1, "frustrated": 1.2,
        "frustrating": 1.2, "irritated": 1.1, "irritating": 1.1, "mad": 1.1,
        "livid": 1.4, "outraged": 1.5, "disgusted": 1.3, "disgusting": 1.3,
        "infuriated": 1.5, "enraged": 1.5, "hostile": 1.2, "bitter": 1.0,
        "resentful": 1.1, "unacceptable": 1.2, "ridiculous": 1.1,
        "pathetic": 1.2, "worst": 1.3, "bad": 0.9, "useless": 1.1,
        "broken": 0.9, "rubbish": 1.1,
    },
    "fear": {
        "scared": 1.3, "afraid": 1.2, "worried": 1.1, "worry": 1.0,
        "anxious": 1.2, "anxiety": 1.2, "terrified": 1.5, "panic": 1.4,
        "panicking": 1.4, "nervous": 1.1, "frightened": 1.3,
        "alarmed": 1.2, "concerned": 0.9, "dread": 1.3, "horror": 1.3,
        "phobia": 1.3, "uneasy": 1.0, "apprehensive": 1.1, "tense": 1.0,
        "stressed": 1.1, "stressful": 1.1, "overwhelmed": 1.2,
        "insecure": 1.1, "threatened": 1.2,
        "panic": 1.4, "panics": 1.3, "panicked": 1.4, "panicking": 1.4,
    },
    "surprise": {
        "wow": 1.2, "whoa": 1.2, "unbelievable": 1.2, "shocked": 1.3,
        "shocking": 1.3, "unexpected": 1.1, "astonished": 1.3,
        "stunned": 1.3, "bewildered": 1.1, "startled": 1.1,
        "speechless": 1.2, "remarkable": 1.0, "extraordinary": 1.1,
        "unprecedented": 1.1, "surprised": 1.2, "surprising": 1.1,
        "suddenly": 0.7, "unreal": 1.0,
        "incredible": 1.2, "astonishing": 1.3, "unforeseen": 1.1,
    },
    "neutral": {
        "okay": 0.7, "ok": 0.6, "fine": 0.7, "regular": 0.6,
        "normal": 0.6, "average": 0.6, "standard": 0.5, "typical": 0.5,
        "ordinary": 0.5, "routine": 0.6, "everyday": 0.5, "usual": 0.5,
    },
}

#: Every lexicon term, flattened; stops a negation scan at an earlier emotion word.
_ALL_TERMS = {term for terms in LEXICON.values() for term in terms}

NEGATORS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "cannot", "cant", "can't", "dont", "don't", "doesnt", "doesn't",
    "didnt", "didn't", "isnt", "isn't", "arent", "aren't", "wasnt",
    "wasn't", "werent", "weren't", "wont", "won't", "wouldnt", "wouldn't",
    "shouldnt", "shouldn't", "couldnt", "couldn't", "hardly", "barely",
    "without", "aint", "ain't",
}

INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "extremely": 1.7, "incredibly": 1.6,
    "absolutely": 1.6, "completely": 1.5, "totally": 1.5, "so": 1.3,
    "quite": 1.2, "super": 1.4, "highly": 1.4, "deeply": 1.4,
    "utterly": 1.6, "truly": 1.3,
}

DIMINISHERS = {
    "slightly": 0.6, "somewhat": 0.7, "mildly": 0.6, "marginally": 0.5,
    "kinda": 0.7, "barely": 0.5,
}

#: Where a negated emotion lands instead.
NEGATION_TARGET = {
    "happy": "sad",
    "sad": "happy",
    "angry": "neutral",
    "fear": "neutral",
    "surprise": "neutral",
    "neutral": "neutral",
}

#: Higher temperature = softer, more cautious probabilities.
TEMPERATURE = 0.6
NEGATION_WINDOW = 3
INTENSIFIER_WINDOW = 2

#: Tokens that end a negation's scope, so "not angry, just disappointed"
#: negates "angry" but leaves "disappointed" intact.
CLAUSE_BOUNDARIES = {
    "but", "just", "and", "because", "although", "though", "however",
    "yet", "so", "while", "whereas", "still",
}

TOKEN_RE = re.compile(r"[a-zA-Z']+")


class SentimentModel:
    """Rule-based emotion scorer with optional scikit-learn override."""

    EMOTIONS = EMOTIONS
    EMOTION_EMOJIS = EMOTION_EMOJIS
    KEYWORDS = LEXICON  # backwards compatible alias

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path or os.path.join("models", "sentiment_model.pkl")
        self.kind = "lexicon"
        self.is_loaded = True
        self._load_model()

    # ------------------------------------------------------------------ setup

    def _load_model(self) -> None:
        """Load an optional scikit-learn pipeline; fall back to the lexicon."""
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, "rb") as handle:
                payload = pickle.load(handle)
            self.model = payload.get("model")
            self.vectorizer = payload.get("vectorizer")
            if self.model is not None and self.vectorizer is not None:
                self.kind = "sklearn"
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Could not load {self.model_path}: {exc}. Using lexicon engine.")

    # ------------------------------------------------------------ preprocessing

    def tokenize(self, text: str) -> List[str]:
        """Lowercase word tokens with punctuation stripped."""
        return TOKEN_RE.findall((text or "").lower().replace("’", "'"))

    def preprocess(self, text: str) -> str:
        """Backwards-compatible helper: normalised token string."""
        return " ".join(self.tokenize(text))

    # ---------------------------------------------------------------- lexicon

    def _lexicon_scores(self, tokens: List[str]):
        """Weighted hits per emotion plus the evidence behind them."""
        scores = {emotion: 0.0 for emotion in self.EMOTIONS}
        signals = []

        for index, token in enumerate(tokens):
            emotion = None
            weight = 0.0
            for candidate, terms in LEXICON.items():
                if token in terms:
                    emotion = candidate
                    weight = terms[token]
                    break
            if emotion is None:
                continue

            multiplier = 1.0
            for offset in range(1, INTENSIFIER_WINDOW + 1):
                previous = tokens[index - offset] if index - offset >= 0 else None
                if previous in INTENSIFIERS:
                    multiplier *= INTENSIFIERS[previous]
                    break
                if previous in DIMINISHERS:
                    multiplier *= DIMINISHERS[previous]
                    break

            window = tokens[max(0, index - NEGATION_WINDOW):index]
            negated = False
            for token_before in reversed(window):
                if token_before in CLAUSE_BOUNDARIES or token_before in _ALL_TERMS:
                    break
                if token_before in NEGATORS:
                    negated = True
                    break

            target = NEGATION_TARGET[emotion] if negated else emotion
            effective = weight * multiplier * (0.8 if negated else 1.0)
            scores[target] += effective
            signals.append(
                {
                    "token": token,
                    "emotion": target,
                    "source_emotion": emotion,
                    "weight": round(effective, 3),
                    "negated": negated,
                }
            )

        return scores, signals

    @staticmethod
    def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
        """Turn raw weights into a probability distribution.

        Every emotion keeps a non-zero mass, so a single weak keyword match
        yields a cautious probability instead of a hard 1.0.
        """
        peak = max(scores.values())
        exps = {
            emotion: math.exp((value - peak) / TEMPERATURE)
            for emotion, value in scores.items()
        }
        total = sum(exps.values())
        return {emotion: value / total for emotion, value in exps.items()}

    def _lexicon_analysis(self, tokens: List[str]) -> Dict:
        """Classify with the lexicon engine."""
        scores, signals = self._lexicon_scores(tokens)

        if not signals:
            probabilities = {emotion: 0.0 for emotion in self.EMOTIONS}
            probabilities["neutral"] = 1.0
            emotion = "neutral"
            confidence = 1.0
        else:
            probabilities = self._softmax(scores)
            emotion = max(probabilities, key=probabilities.get)
            confidence = probabilities[emotion]

        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "scores": {key: round(value, 3) for key, value in probabilities.items()},
            "signals": signals,
            "engine": "lexicon",
        }

    # --------------------------------------------------------------------- ml

    def _ml_analysis(self, text: str) -> Dict:
        """Classify with a scikit-learn pipeline when one is available."""
        if self.model is None or self.vectorizer is None:
            return self._lexicon_analysis(self.tokenize(text))

        try:
            features = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(features)[0]
            classes = [str(label) for label in self.model.classes_]
            scores = {emotion: 0.0 for emotion in self.EMOTIONS}
            for label, probability in zip(classes, probabilities):
                if label in scores:
                    scores[label] = float(probability)
            emotion = max(scores, key=scores.get)
            return {
                "emotion": emotion,
                "confidence": round(scores[emotion], 3),
                "scores": {key: round(value, 3) for key, value in scores.items()},
                "signals": [],
                "engine": "sklearn",
            }
        except Exception as exc:  # pragma: no cover - defensive
            print(f"ML analysis failed: {exc}. Falling back to lexicon.")
            return self._lexicon_analysis(self.tokenize(text))

    # ------------------------------------------------------------------ public

    def analyze(self, text: str) -> Dict:
        """Score a single piece of text."""
        if self.model is not None:
            result = self._ml_analysis(text)
        else:
            result = self._lexicon_analysis(self.tokenize(text))
        result["emoji"] = self.EMOTION_EMOJIS.get(result["emotion"], "❓")
        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Score many texts."""
        return [self.analyze(text) for text in texts]

    def get_distribution(self, texts: List[str]) -> Dict:
        """Emotion counts across a corpus."""
        results = self.analyze_batch(texts)
        counts = Counter(result["emotion"] for result in results)
        total = len(results) or 1
        return {
            emotion: {
                "count": counts.get(emotion, 0),
                "percentage": round(counts.get(emotion, 0) / total * 100, 2),
            }
            for emotion in self.EMOTIONS
        }


# Singleton instance
_model_instance = None


def get_model() -> SentimentModel:
    """Get or create the shared model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentModel()
    return _model_instance
