import spacy

nlp = spacy.load("../model/model-best")

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return "This is NER model app............"

@app.route('/predict', methods=['POST'])
def predict():
    # extract input data from request
    data = request.get_json()

    # extract text from input data
    text = data['text']

    # use spaCy NER model to generate predictions
    doc = nlp(text)

    # extract named entities from doc
    entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

    # return predictions as JSON
    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True)



