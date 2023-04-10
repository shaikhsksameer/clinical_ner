import spacy
import json
import pytest
from evaluation.confusion_matrix import calculate_entity_metrics

best_nlp = spacy.load("../model/model-best")

with open("../data/Corona2.json", "r") as f:
    data = json.load(f)


def test_calculate_entity_metrics():
    # Test with empty data and trained best_nlp model
    test_data = {"examples": []}
    assert calculate_entity_metrics(test_data, best_nlp) == {"PATHOGEN": {"count": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0},
                                                        "MEDICINE": {"count": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0},
                                                        "MEDICALCONDITION": {"count": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}}


    # Test with incorrect entity type
    data = {"examples": [{"content": "I have a headache", "annotations": [{"start": 9, "end": 18, "tag_name": "invalid_type"}]}]}
    with pytest.raises(KeyError):
        assert calculate_entity_metrics(data, best_nlp) == {"PATHOGEN": {"count": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 1.0},
                                                        "MEDICINE": {"count": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 1.0},
                                                        "MEDICALCONDITION": {"count": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 1.0}}
