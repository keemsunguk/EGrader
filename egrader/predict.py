import os
import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer

'''
Prediction

For prediction, we need to prepare the input text the same way as we did for training -
tokenize, adding the special [CLS] and [SEP] token at begin and end of the token sequence,
and pad to match the model input shape.
'''


def predict_movie_review(model, data, bert_ckpt_dir, pred_sentences):
    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
    pred_tokens    = map(tokenizer.tokenize, pred_sentences)
    pred_tokens    = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))

    print('pred_token_ids', pred_token_ids.shape)

    res = model.predict(pred_token_ids).argmax(axis=-1)
    return res


def predict_essay_grade(model, data, bert_ckpt_dir, pred_sentences):
    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
    pred_tokens    = map(tokenizer.tokenize, pred_sentences)
    pred_tokens    = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))

    print('pred_token_ids', pred_token_ids.shape)

    res = model.predict(pred_token_ids).argmax(axis=-1)
    return res
