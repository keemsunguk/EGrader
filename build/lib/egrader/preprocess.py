from bs4 import BeautifulSoup
import spacy

model = 'en_core_web_lg'

class Preprocess:
    """
    Preprocess base class
    """
    def __init__(self, raw_obj):
        self.raw_obj = raw_obj
        self.soup = None
        self.text = None
        self.doc = None
        self.nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)

    def clean_html(self, obj=None):
        if obj:
            self.raw_obj = obj
        self.soup = BeautifulSoup(self.raw_obj, 'html.parser')
        self.text = self.soup.getText()
        return self.text

    def make_doc_obj(self, obj=None):
        if obj:
            self.text = obj
        self.doc = self.nlp(self.text)
        return self.doc
