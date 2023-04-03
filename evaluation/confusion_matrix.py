import spacy
import json

best_nlp = spacy.load("../model/model-best")

with open("../data/Corona2.json", "r") as f:
    data = json.load(f)

entity_counts = {"PATHOGEN": 0, "MEDICINE": 0, "MEDICALCONDITION": 0}
entity_metrics = {"PATHOGEN": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                  "MEDICINE": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                  "MEDICALCONDITION": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}

for example in data["examples"]:
    text = example["content"]
    doc = best_nlp(text)

    # Get the predicted entities and their labels
    predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

    # Get the true entities and their labels
    true_entities = [(annot["start"], annot["end"], annot["tag_name"].upper()) for annot in example["annotations"]]

    # Compute the number of true positives, false positives, false negatives, and true negatives for each entity
    for true_entity in true_entities:
        entity_type = true_entity[2]
        entity_counts[entity_type] += 1
        if true_entity in predicted_entities:
            entity_metrics[entity_type]["tp"] += 1
        else:
            entity_metrics[entity_type]["fn"] += 1

    for predicted_entity in predicted_entities:
        entity_type = predicted_entity[2]
        if predicted_entity not in true_entities:
            entity_metrics[entity_type]["fp"] += 1
        else:
            entity_metrics[entity_type]["tn"] += 1

# Compute precision, recall, F1 score, true negatives, and accuracy for each entity
for entity_type, metrics in entity_metrics.items():
    tp = metrics["tp"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    tn = metrics["tn"]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    count = entity_counts[entity_type]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("Entity type:", entity_type)
    print("Count:", count)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1_score)
    print("Accuracy:", accuracy)
    print()
