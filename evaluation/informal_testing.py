import spacy
from  spacy import displacy

nlp_ner = spacy.load("../model/model-best")

test_text = "While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present."

doc = nlp_ner(test_text)

colors = {"PATHOGEN": "#F67DE3", "MEDICINE": "#7DF6D9", "MEDICALCONDITION":"#a6e22d"}
options = {"colors": colors}

for ent in doc.ents:
    print(ent.text , ent.label_)

spacy.displacy.serve(doc, style="ent", options= options , auto_select_port=True)

