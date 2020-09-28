import tensorflow as tf
import pandas as pd
import sys
import os
import re
import pickle
import json

from bert.tokenization.bert_tokenization import FullTokenizer

from egrader.data_util import load_dataset, EssayGraderData
from egrader.bert_util import create_model
from egrader.train import train_essay_grader
from egrader.predict import predict_essay_grade
from egrader.data_util import load_essay_from_db
from egrader.db_util import DBUtil, preprocess

model_config = {
    'bert_ckpt_dir': '/Users/keemsunguk/Projects/data/uncased_L-2_H-128_A-2',
    'bert_ckpt_file': '/Users/keemsunguk/Projects/data/uncased_L-2_H-128_A-2/bert_model.ckpt',
    'bert_config_file': '/Users/keemsunguk/Projects/data/uncased_L-2_H-128_A-2/bert_config.json'
}


def run(data_df, bert_ckpt_dir, bert_config_file, bert_ckpt_file, output_model):
    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
    data = EssayGraderData(tokenizer, data_df, sample_size=10*128*2, max_seq_len=512)

    print("            train_x", data.train_x.shape)
    print("train_x_token_types", data.train_x_token_types.shape)
    print("            train_y", data.train_y.shape)
    print("             test_x", data.test_x.shape)
    print("        max_seq_len", data.max_seq_len)

    adapter_size = None  # use None to fine-tune all of BERT
    model = create_model(data.max_seq_len, bert_config_file, bert_ckpt_file, adapter_size=adapter_size)
    model = train_essay_grader(model, data)
    model.save_weights(output_model, overwrite=True)


def main():
    bert_ckpt_dir = model_config['bert_ckpt_dir']
    bert_ckpt_file = model_config['bert_ckpt_file']
    bert_config_file = model_config['bert_config_file']

# Uncomment this block to read inputs from json
#    with open('/Users/keemsunguk/Projects/EssayGrader/data/toefl_list.json', 'r') as rf:
#        toefl_list = json.load(rf)
#    toefl_df = pd.DataFrame(toefl_list, columns=['essay', 'rating'])

    with open('/Users/keemsunguk/Projects/data/toefl_balanced.pkl', 'rb') as rf:
        toefl_df = pd.read_pickle(rf)
    toefl_df['rating'] = toefl_df.apply(lambda x: round(x['rating']), axis=1)
#    toefl_df['essay'] = toefl_df.apply(lambda x: preprocess(x['essay'], mask_number=False, mask_unknown=True), axis=1)

    output_model = '/Users/keemsunguk/Projects/data/trained_essay_model_L2_H128_A2.h5'
    run(toefl_df, bert_ckpt_dir, bert_config_file, bert_ckpt_file, output_model)


if __name__ == "__main__":
    main()