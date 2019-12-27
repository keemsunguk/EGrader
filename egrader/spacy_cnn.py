import random
from pathlib import Path
import spacy
import logging
from spacy.util import minibatch, compounding
from egrader.db_util import DBUtil, config

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

db_util = DBUtil()
conf = config.get_config()

options = {
    'model': "en_core_web_lg",
    'output_dir': conf['Projects']+conf['EGraderRoot']+"data",
    'n_texts': 3000,
    'n_iter': 20,
    'init_tok2vec': None
}


def prepare_output_dir(output_dir):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
    pass


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        predicted = [[l, s] for l, s in doc.cats.items()]
        labels, scores = zip(*predicted)
        p = labels[scores.index(max(scores))]

        if gold[p]:
            tp += 1.0
        else:
            fp += 1
    accuracy = tp / (tp + fp)
    return {"textcat_a": accuracy}


class SpacyCNN:
    def __init__(self, test_type, options=options):
        if test_type not in ['SAT', 'GRE', 'TOEFL']:
            logger.error('Unavailable Test Type: %s', test_type)
            return False
        model = options['model']
        prepare_output_dir(options['output_dir'])
        self.n_texts = options['n_texts']
        self.n_iter = options['n_iter']
        self.train_data = None
        self.dev_texts = None
        self.dev_cats = None
        self.optimizer = None
        self.nlp2 = None
        if model is not None:
            self.nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
        else:
            self.nlp = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")
        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "textcat" not in self.nlp.pipe_names:
            self.textcat = self.nlp.create_pipe(
                "textcat", config={"exclusive_classes": True, "architecture": "ensemble"}
            )
            self.nlp.add_pipe(self.textcat, last=True)
        # otherwise, get it, so we can add labels to it
        else:
            self.textcat = self.nlp.get_pipe("textcat")

    def add_text_category(self, cats):
        # add label to text classifier
        for cat in cats:
            self.textcat.add_label(cat)
        pass

    def load_df(self, test_type):
        logger.info("Loading %s data...", test_type)
        df = db_util.get_spacy_labeled_essays(test_type, merge_0_1=True)
        options['n_texts'] = df.shape[0]
        n_texts = options['n_texts']
        (train_texts, train_cats), (dev_texts, dev_cats) = self.format_spacy_data(df, split=0.9)
        self.dev_texts = dev_texts
        self.dev_cats = dev_cats
        print(
            "Using {} examples ({} training, {} evaluation)".format(
                n_texts, len(train_texts), len(dev_texts)
            )
        )
        self.train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    def format_spacy_data(self, df, limit=0, split=0.8):
        def make_label_dict(r):
            rdict = {'0': False, '1': False, '2': False, '3': False, '4': False, '5': False, '6': False, r: True}
            return rdict

        # Partition off part of the train data for evaluation
        train_data = [(v[0], v[1]) for k, v in df.iterrows()]
        random.shuffle(train_data)
        train_data = train_data[-limit:]
        texts, rate = zip(*train_data)
        labels = [make_label_dict(x) for x in rate]
        split = int(len(train_data) * split)
        return (texts[:split], labels[:split]), (texts[split:], labels[split:])

    def train_text(self, train_data=None):
        if not train_data:
            train_data = self.train_data
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "textcat"]
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            self.optimizer = self.nlp.begin_training()
            if options['init_tok2vec'] is not None:
                with options['init_tok2vec'].open("rb") as file_:
                    self.textcat.model.tok2vec.from_bytes(file_.read())
            print("Training the model...")
            print("{:^5}\t{:^5}".format("LOSS", "Accuracy"))
            batch_sizes = compounding(4.0, 64.0, 1.001)
            for i in range(self.n_iter):
                losses = {}
                # batch up the examples using spaCy's minibatch
                random.shuffle(train_data)
                batches = minibatch(train_data, size=batch_sizes)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(texts, annotations, sgd=self.optimizer, drop=0.3, losses=losses)
                with self.textcat.model.use_params(self.optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = evaluate(
                        self.nlp.tokenizer,
                        self.textcat,
                        self.dev_texts,
                        self.dev_cats)
                print(
                    "{0:.3f}\t{1:.3f}".format(  # print a simple table
                        losses["textcat"],
                        scores["textcat_a"],
                    )
                )

    def classify_text(self, test_text):
        doc = self.nlp(test_text)
        print(doc.cats)
        print(test_text)
        pass

    def store_model(self, output_dir):
        if output_dir is not None:
            with self.nlp.use_params(self.optimizer.averages):
                self.nlp.to_disk(output_dir)
            print("Saved model to", output_dir)
        pass

    def load_model(self, model_dir):
        print("Loading from", model_dir)
        self.nlp2 = spacy.load(model_dir)
        pass
