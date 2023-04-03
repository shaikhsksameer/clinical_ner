import spacy
import json

best_nlp = spacy.load("../model/model-best")  # Load the trained model



with open("../data/Corona2.json", "r") as f:
    data = json.load(f)

true_positives = 0
false_positives = 0
false_negatives = 0

for example in data["examples"]:
    text = example["content"]
    doc = best_nlp(text)  # Process the text with the model

    # Get the predicted entities and their labels
    predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

    # Get the true entities and their labels
    true_entities = [(annot["start"], annot["end"], annot["tag_name"].upper()) for annot in example["annotations"]]


    # Compute the number of true positives, false positives, and false negatives
    for entity in predicted_entities:
        if entity in true_entities:
            true_positives += 1
        else:
            false_positives += 1
    for entity in true_entities:
        if entity not in predicted_entities:
            false_negatives += 1

print("True Positive :",true_positives)
print("False Positive :",false_positives)
print("False Negative :",false_negatives)

# Compute precision, recall, and F1 score
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)
