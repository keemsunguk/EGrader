import os
import logging
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()
import tensorflow as tf
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
from egrader.db_util import DBUtil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(__name__)

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["label"] = []
    for file_path in tqdm(os.listdir(directory), desc=os.path.basename(directory)):
        with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["label"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    return train_df, test_df


def copy_and_load_dataset():
    data_dir = '/Users/keemsunguk/Projects/data/aclImdb/'
    train = load_dataset(data_dir + 'train')
    test = load_dataset(data_dir + 'test')
    return train, test


def load_essay_from_db():
    edb = DBUtil()
    toefls = edb.get_labeled_essays('TOEFL', merge_0_1=True, with_topic=True, mask_number=False)
    toefls.columns = ['essay', 'rating']
    logger.info(str(len(toefls)))
    print('Total data:', len(toefls))
    train, test = train_test_split(toefls, train_size=0.7)
    return train, test


def load_essay_from_dataframe(essay_df):
    essay_df.columns = ['essay', 'rating']
    logger.info(str(len(essay_df)))
    print('Total data:', len(essay_df))
    train, test = train_test_split(essay_df, train_size=0.7)
    return train, test


class MovieReviewData:
    DATA_COLUMN = "sentence"
    LABEL_COLUMN = "polarity"

    def __init__(self, tokenizer: FullTokenizer, sample_size=None, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        train, test = copy_and_load_dataset()  # download_and_load_datasets()

        train, test = map(lambda df: df.reindex(df[MovieReviewData.DATA_COLUMN].str.len().sort_values().index),
                          [train, test])

        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test = train.head(sample_size), test.head(sample_size)
            # train, test = map(lambda df: df.sample(sample_size), [train, test])

        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad,
                                                       [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[MovieReviewData.DATA_COLUMN], row[MovieReviewData.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)
                y.append(int(label))
                pbar.update()
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)


class EssayGraderData:
    DATA_COLUMN = "essay"
    LABEL_COLUMN = "rating"

    def __init__(self, tokenizer: FullTokenizer, essay_df=None, sample_size=None, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        if essay_df is not None:
            train, test = load_essay_from_dataframe(essay_df)
        else:
            train, test = load_essay_from_db()

        train, test = map(lambda df: df.reindex(df[EssayGraderData.DATA_COLUMN].str.len().sort_values().index),
                          [train, test])

        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test = train.head(sample_size), test.head(sample_size)
            # train, test = map(lambda df: df.sample(sample_size), [train, test])

        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad,
                                                       [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[EssayGraderData.DATA_COLUMN], row[EssayGraderData.LABEL_COLUMN]
                try:
                    tokens = self.tokenizer.tokenize(text)
                    tokens = ["[CLS]"] + tokens + ["[SEP]"]
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    self.max_seq_len = max(self.max_seq_len, len(token_ids))
                    x.append(token_ids)
                    y.append(int(label))
                    pbar.update()
                except:
                    # print(tokens)
                    pass
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)
