from bs4 import BeautifulSoup
import spacy
import re

model = 'en_core_web_lg'


def clean_abbreviation(txt):
    ptrn = r'(?:[A-Z]\.)+'
    for f in re.finditer(ptrn, txt):
        old_ptrn = f.group()
        new_ptrn = old_ptrn.replace('.', '')
        txt = txt.replace(old_ptrn, new_ptrn)

    return txt


class Preprocess:
    """
    Preprocess Class accepts HTML text as an input and cleans and process it for a various format.
    Example input is:
        nyt = requests.get(nyt_url+"/2020/08/01/opinion/sunday/mail-voting-covid-2020-election.html")
        Preprocess(nyt.text)
    """
    def __init__(self, raw_obj = None):
        """
        Preporcess initializer
        :param raw_obj: HTML text i.e., request.text
        """
        self.raw_obj = raw_obj
        self.soup = None
        self.text = None
        self.doc = None
        self.nlp = None

    def load_spacy_lg(self):
        """
        Load Spacy en_core_web_lg model
        :return: None
        """
        self.nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
        return None

    def clean_html(self, obj=None):
        """
        Remove tags and return text in the html
        :param obj: HTML text i.e., request.text
        :return: HTML text
        """
        if obj:
            self.raw_obj = obj
        self.soup = BeautifulSoup(self.raw_obj, 'html.parser')
        self.text = self.soup.getText()
        return self.text

    def make_doc_obj(self, obj: str = None):
        """
        Make a Spacy doc object from the text
        :param obj: Text as string
        :return: Spacy Doc Object
        """
        if obj:
            self.text = obj
        self.load_spacy_lg()
        self.doc = self.nlp(self.text)
        return self.doc

    def extract_body_sentences(self, obj=None) -> list:
        """
        Extract Body text.  It is removing dots from abbreviations for better sentence-izing
        :param obj: HTML Text
        :return: List of sentences
        """
        if obj:
            self.raw_obj = obj
        self.soup = BeautifulSoup(self.raw_obj, "html.parser")
        txt_body = self.soup.find('body')
        sentences = []
        skip_dot = ['Mr.', 'Mrs.', 'Dr.', 'Jan.', 'Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.',
                    'Nov.', 'Dec.']
        for pt in txt_body.find_all('p'):
            txt = pt.text
            if '.' in txt:
                for sdot in skip_dot:
                    txt = txt.replace(sdot, sdot[:-1])
                txt = clean_abbreviation(txt)
                tmp_sent = txt.split('.')
                sentences += [s.strip()+'.' for s in tmp_sent if len(s) > 0]
            else:
                if '”' == txt or '"' == txt:
                    sentences[-1] += txt
                else:
                    sentences.append(txt)
            sentences.append('\n')
        return sentences
