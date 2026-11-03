"""Run the labelled sanity set and print accuracy plus a confusion table.

Usage:
    cd backend && python -m eval.run_eval
"""

from collections import Counter

from ml.model import SentimentModel, EMOTIONS
from eval.emotion_eval import DATASET


def run() -> float:
    model = SentimentModel(model_path="__none__")
    correct = 0
    confusion = Counter()

    for text, expected in DATASET:
        predicted = model.analyze(text)["emotion"]
        confusion[(expected, predicted)] += 1
        correct += predicted == expected

    accuracy = correct / len(DATASET)
    print(f"accuracy: {correct}/{len(DATASET)} = {accuracy:.1%}\n")

    header = "actual \\ predicted"
    print(f"{header:22}{''.join(f'{e[:8]:>10}' for e in EMOTIONS)}")
    for expected in EMOTIONS:
        row = "".join(f"{confusion[(expected, p)] or '':>10}" for p in EMOTIONS)
        print(f"{expected:22}{row}")

    misses = [
        (text, expected, model.analyze(text)["emotion"])
        for text, expected in DATASET
        if model.analyze(text)["emotion"] != expected
    ]
    if misses:
        print("\nmisclassified:")
        for text, expected, predicted in misses:
            print(f"  {expected:9} -> {predicted:9} {text}")
    return accuracy


if __name__ == "__main__":
    run()
