#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier
* Training: https://spacy.io/usage/training
Compatible with: spaCy v2.0.0
"""
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from egrader.db_util import DBUtil

db_util = DBUtil()

options = {
    'model': "en_core_web_lg",
    'output_dir': "/Users/keemsunguk/Projects/EssayGrader/data",
    'n_texts': 3000,
    'n_iter': 20,
    'init_tok2vec': None
}

def main():
    output_dir = options['output_dir']
    model = options['model']
    n_texts = options['n_texts']
    n_iter = options['n_iter']

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "ensemble"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    textcat.add_label("1")
#    textcat.add_label("2")
#    textcat.add_label("3")
#    textcat.add_label("4")
#    textcat.add_label("5")
    textcat.add_label("6")

    # load the SAT dataset
    print("Loading SAT data...")
    sat_df = db_util.get_spacy_labeled_essays('SAT', merge_0_1=True)
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(sat_df)
    train_texts = train_texts[:n_texts]
    train_cats = train_cats[:n_texts]
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            n_texts, len(train_texts), len(dev_texts)
        )
    )
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if options['init_tok2vec'] is not None:
            with options['init_tok2vec'].open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    # test the trained model
    test_text = "This essay sucked"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)


def load_data(df, limit=0, split=0.8):
    # Partition off part of the train data for evaluation
    train_data = [(v[0], v[1]) for k, v in df.iterrows()]
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, rate = zip(*train_data)
    labels = [{"1": True, "6": False} if r == "1" else {"1": False, "6": True} for r in rate]
    split = int(len(train_data) * split)
    return (texts[:split], labels[:split]), (texts[split:], labels[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "1":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    main()

class SpacyCNN:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
