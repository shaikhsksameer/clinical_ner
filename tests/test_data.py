import pytest
from spacy.tokens import DocBin
import json
import jsonschema
from src.model import convert_to_old_spacy_format
from src.model import create_training_data


def test_load_training_data():
    """
    This test checks that the JSON file is successfully loaded and that the examples key in the JSON object contains at least one example.
    :return:
    """
    with open("../data/Corona2.json", "r") as f:
        data = json.load(f)
    assert len(data['examples']) > 0


"""
pytest.fixture is a decorator provided by the pytest testing framework in Python. It is used to define a fixture, 
which is a function that provides a fixed set of data or objects that can be reused across multiple tests.
"""
@pytest.fixture
def data():
    # Define sample data for testing
    data = {
        "examples": [
            {
                "content": "While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.[92]\n\nDiosmectite, a natural aluminomagnesium silicate clay, is effective in alleviating symptoms of acute diarrhea in children,[93] and also has some effects in chronic functional diarrhea, radiation-induced diarrhea, and chemotherapy-induced diarrhea.[45] Another absorbent agent used for the treatment of mild diarrhea is kaopectate.\n\nRacecadotril an antisecretory medication may be used to treat diarrhea in children and adults.[86] It has better tolerability than loperamide, as it causes less constipation and flatulence.[94]",
                "annotations": [
                    {
                        "start": 360,
                        "end": 371,
                        "tag_name": "Medicine"
                    }
                ]
            }
        ]
    }
    return data

def test_convert_to_old_spacy_format(data):
    # Load the JSON data to be validated
    # Call the function with the sample data
    data = convert_to_old_spacy_format(data)

schema = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string"
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "integer"
                    },
                    "end": {
                        "type": "integer"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["MEDICINE", "MEDICALCONDITION"]
                    }
                },
                "required": ["start", "end", "type"]
            }
        }
    },
    "required": ["text", "entities"]
}
# Validate the JSON data against the schema
try:
    jsonschema.validate(instance=data, schema=schema)
    print("Data is valid")
except jsonschema.exceptions.ValidationError as err:
    print(err)


# Negative test case 1: Missing 'examples' key
def test_missing_examples_key():
    data = {"sample": "data"}
    with pytest.raises(KeyError):
        convert_to_old_spacy_format(data)


# Negative test case 2: Missing 'content' or 'annotations' keys
def test_missing_annotations_key():
    data = {"examples": [{ "content":"While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements."}]}
    with pytest.raises(TypeError):
        convert_to_old_spacy_format(data)



# Negative test case 3: Missing 'start', 'end', or 'tag_name' keys in annotations
def test_missing_keys_in_annotations():
    invalid_data = {"examples": [{"content": "While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements.", "annotations": [{"start": 360, "end": 371}]}]}
    assert convert_to_old_spacy_format(invalid_data) == []


# Test case for invalid start or end values in annotations
def test_invalid_start_or_end_values():
    invalid_example = {"content": "invalid values example", "annotations": [{"start": -1, "end": 5, "tag_name": "Medicine"}]}
    invalid_data = {"examples": [invalid_example]}
    assert convert_to_old_spacy_format(invalid_data) == []


# Test case for invalid label values in entities
def test_invalid_label_values():
    invalid_example = {"content": "invalid label example", "annotations": [{"start": 0, "end": 5, "tag_name": "invalid_label"}]}
    invalid_data = {"examples": [invalid_example]}
    assert convert_to_old_spacy_format(invalid_data) == []



@pytest.fixture
def training_data():
    return [("This is a sentence", {'entities': [(0, 4, 'LABEL')]})]

def test_create_training_data(training_data):
    db = create_training_data(training_data)
    assert isinstance(db, DocBin)
    assert len(db) == 1




