
# Named Entity recognition (NER) in spaCy
Custom Named Entity Recognition Model using Spacy to extract The required  entities from the given text.

This repository contains code for building a custom NER model using spaCy. The model is trained on data in the old spaCy format, converted from JSON format. The training and testing data are saved in the new spaCy format for training the model. The model is trained using the spaCy CLI, and evaluated on test data using the same CLI


## Acknowledgements

- [spaCy's NER model](https://spacy.io/universe/project/video-spacys-ner-model)
 - [Using spaCy 3.0 to build a custom NER model](https://towardsdatascience.com/using-spacy-3-0-to-build-a-custom-ner-model-c9256bea098)
 - [Custom Named Entity Recognition](https://medium.com/analytics-vidhya/custom-named-entity-recognition-ner-model-with-spacy-3-in-four-steps-7e903688d51)
 


## Setup
Before running the code, you will need to install the following dependencies:
    Step 1 - Install Spacy using pip command

    pip install spacy
   Step 2 - Download best matching default model

    python -m spacy download en


## Data

The training and testing data are stored in the ../data/ directory. The training data is stored in train.spacy and the testing data is stored in test.spacy. Both files are in the new spaCy format, which is a binary format.

The data is originally in JSON format, and is loaded using the json module. Each document in the JSON data is a dictionary with two keys: content and annotations. The content key contains the text of the document, and the annotations key contains a list of entities in the document.

The entities in the annotations key are dictionaries with three keys: start, end, and tag_name. The start key contains the starting character index of the entity, the end key contains the ending character index of the entity, and the tag_name key contains the label of the entity.

The code then converts this data to the old spaCy format, which is a list of tuples. Each tuple contains the text of the document and a dictionary with one key: entities. The entities key contains a list of entities, where each entity is a tuple with three elements: the starting character index, the ending character index, and the label of the entity.

The old spaCy format is then converted to the new spaCy format using the DocBin class from the spaCy library.
## Training
The model is trained using the spaCy CLI. The configuration file for the model is created using the spacy init fill-config command, which takes the base_config.cfg file as input and creates a new configuration file config.cfg.

    Creating a config file using Spacy CLI
    python -m spacy init fill-config base_config.cfg config.cfg
    
The config.cfg file specifies the parameters for training the model, such as the dropout rate, the number of iterations, and the batch size.

The model is trained using the spacy train command, which takes the configuration file as input and saves the trained model to the ../model/output directory.

    train the model using CLI
    python -m spacy train config.cfg --output ../model
`

## Testing
The model is evaluated on test data using the spaCy CLI. The spacy evaluate command takes the path to the trained model and the path to the test data as input, and prints out the evaluation metrics, such as precision, recall, and F1 score.

    to evaluate the model on test data using spacy CLI
    python -m spacy evaluate ../model/output/model-best ../data/train.spacy