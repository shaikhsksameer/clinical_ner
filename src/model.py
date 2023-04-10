import spacy
import json
from spacy.tokens import DocBin
from spacy.util import filter_spans

nlp = spacy.load("en_core_web_sm")

with open("../data/Corona2.json" , "r") as f:
    data = json.load(f)

# print(data)
# print(type(data))
# print(len(data['examples']))
# print(data['examples'][0])


#
# for example in data['examples']:
#     text = example['content']
#     print(text)


"""
    to convert the data in old spacy format i.e. annoteted format [('text', {'entities':[(start,end,label),()]}]
 """

def convert_to_old_spacy_format(data):
    training_data = []
    VALID_LABELS = ["MEDICINE", "PATHOGEN", "MEDICALCONDITION"]
    for document in data['examples']:
        text = document.get('content')
        entities = []
        skip_document = False
        for annot_obj in document.get('annotations'):
            start = annot_obj['start']
            end = annot_obj['end']
            if start < 0 or end > len(text):
                # Skip invalid annotations and respective text
                skip_document = True
                break
            try:
                label = annot_obj['tag_name'].upper()
                if label not in VALID_LABELS:
                    # Skip invalid labels
                    skip_document = True
                    break
            except KeyError:
                raise KeyError("Annotation dictionary is missing the 'tag_name' key")
            entities.append((start, end, label))
        if not skip_document:
            training_data.append((text, {"entities": entities}))
    return training_data


training_data = convert_to_old_spacy_format(data)

print(training_data[0])

# print(len(training_data))


# to create the data in spacy new format for training
def create_training_data(data):
    db = DocBin()
    for text, annotations in data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations.get('entities'):
            span = doc.char_span(start, end, label=label , alignment_mode='contract')
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)

        if not (None in ents):
            filtered = filter_spans(ents)  # THIS DOES THE TRICK for overlapping span
            doc.ents = filtered
            db.add(doc)
    return db


train_data = create_training_data(training_data[:25])
train_data.to_disk("../data/train.spacy")

test_data = create_training_data(training_data[25:])
test_data.to_disk("../data/test.spacy")

"""
creating a config file using spacy CLI
"""
"""
to create config.cfg file
python -m spacy init fill-config base_config.cfg config.cfg
"""
"""
train the model using CLI
python -m spacy train config.cfg --output ../model
"""
"""
to evaluate the model on test data using spacy CLI
python -m spacy evaluate ../model/model-best ../data/train.spacy
"""
