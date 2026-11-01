"""Tests for the emotion engine.

These lock in the behaviour that the old keyword matcher got wrong:
substring false positives, negation, intensity and explainability.
"""

import pytest

from ml.model import SentimentModel
from eval.emotion_eval import DATASET


@pytest.fixture(scope="module")
def model():
    return SentimentModel(model_path="__missing__")


def test_substring_matches_are_not_emotion(model):
    """'made' must not match 'mad' and 'missing' must not match 'miss'."""
    assert model.analyze("I made a decision about the deployment")["emotion"] == "neutral"
    assert model.analyze("The missing replica is back online")["emotion"] == "neutral"


def test_profanity_free_positive(model):
    result = model.analyze("I absolutely love this product, it is amazing")
    assert result["emotion"] == "happy"
    assert result["confidence"] > 0.8


def test_negation_flips_emotion(model):
    assert model.analyze("I am not happy with this result")["emotion"] == "sad"


def test_negation_stops_at_clause_boundary(model):
    """"not angry" should not swallow "disappointed" in the next clause."""
    assert model.analyze("I am not angry, just disappointed")["emotion"] == "sad"


def test_intensifier_increases_confidence(model):
    plain = model.analyze("This is good")["confidence"]
    intense = model.analyze("This is extremely good")["confidence"]
    assert intense > plain


def test_probabilities_sum_to_one(model):
    result = model.analyze("I am worried about the exam")
    assert pytest.approx(sum(result["scores"].values()), abs=1e-2) == 1.0


def test_scores_cover_every_emotion(model):
    scores = model.analyze("wow")["scores"]
    assert set(scores) == set(SentimentModel.EMOTIONS)


def test_signals_explain_prediction(model):
    result = model.analyze("I am terrified of the deadline")
    tokens = [signal["token"] for signal in result["signals"]]
    assert "terrified" in tokens
    assert result["engine"] == "lexicon"


def test_empty_and_plain_text_are_neutral(model):
    assert model.analyze("")["emotion"] == "neutral"
    assert model.analyze("The config file lives in the root folder")["emotion"] == "neutral"


def test_batch_matches_single(model):
    texts = ["I love this", "I hate this", "the build finished"]
    batch = model.analyze_batch(texts)
    assert [item["emotion"] for item in batch] == [model.analyze(t)["emotion"] for t in texts]


def test_distribution_counts_all_emotions(model):
    distribution = model.get_distribution(["I love this", "I hate this"])
    assert set(distribution) == set(SentimentModel.EMOTIONS)
    assert distribution["happy"]["count"] == 1
    assert distribution["angry"]["count"] == 1


def test_accuracy_floor(model):
    """Guard rail: accuracy on the labelled sanity set must not regress."""
    correct = sum(
        model.analyze(text)["emotion"] == expected for text, expected in DATASET
    )
    assert correct / len(DATASET) >= 0.85
