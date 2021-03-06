from pymongo import MongoClient
import pandas as pd
from egrader.config import config, PROJECT_ROOT
import re

conf = config.get_config()
REMOTE_MONGO = conf['RemoteMongo']
LOCAL_MONGO = conf['LocalMongo']

try:
    with open(PROJECT_ROOT+'data/dictionary.txt', 'r') as rf:
        dictionary = rf.readlines()
except:
        dictionary = None


def preprocess(text, mask_number=True, mask_unknown=False):
    return_text = text.lower().replace('\t', ' ').replace('\r', '').replace('\n', ' ')
    if mask_number:
        return_text = re.sub("\s\d+\s", " <NUM> ", return_text)
    if mask_unknown:
        return_text = return_text.replace('  ', ' ')
        tokens = return_text.split(' ')
        if dictionary:
            for i, t in enumerate(tokens):
                if t not in dictionary:
                    tokens[i] = '<UKN>'
            return_text = ' '.join(tokens)
    return return_text


class DBUtil:
    def __init__(self, local_db=False):
        self.essay_collection = None
        self.edb = None
        if local_db:
            local_client = MongoClient(LOCAL_MONGO)
            self.edb = local_client['essay']
            self.essay_collection = self.edb['essay']
        else:
            remote_client = MongoClient(REMOTE_MONGO)
            self.edb = remote_client['nlp']
            self.essay_collection = self.edb['essays']


    def describe_db(self):
        try:
            print('==== Database Description ====')
            print(self.edb.list_collection_names())
            print('Total Essay:', self.essay_collection.count_documents({}))
            print('Total SAT:', self.essay_collection.count_documents({'$and': [{'type': 'SAT'}, {'rate': {'$gte': 0}}]}))
            print('Total TOEFL:', self.essay_collection.count_documents({'$and': [{'type': 'TOEFL'}, {'rate': {'$gt': 0}}]}))
            print('Total GRE:', self.essay_collection.count_documents({'$and': [{'type': 'GRE'}, {'rate': {'$gte': 0}}]}))
        except Exception as e:
            print("No Mongo connection found: %s", str(e))

    def get_sat_essays(self):
        return [e for e in self.essay_collection.find({'$and': [{'type': 'SAT'}, {'rate': {'$gte': 0}}]})
                if len(e['essay']) > 100]

    def get_toefl_essays(self):
        return [e for e in self.essay_collection.find({'$and': [{'type': 'TOEFL'}, {'rate': {'$gt': 0}}]})
                if len(e['essay']) > 100]

    def get_gre_essays(self):
        return [e for e in self.essay_collection.find({'$and': [{'type': 'GRE'}, {'rate': {'$gte': 0}}]})
                if type(e['essay']) == str and len(e['essay']) > 100]

    def get_labeled_essays(self, test_type, merge_0_1=True, with_topic=False, mask_number=True, mask_unknown=False):
        if with_topic:
            df = pd.DataFrame(
                [[e['topic']+'.\r\n'+e['essay'], e['rate']] for e in self.essay_collection.find(
                    {'$and': [{'type': test_type}, {'rate': {'$gte': 0}}, {"$expr": {"$gt": [{"$strLenCP": "$essay"}, 100]}}]})
                    if '강좌 1' not in e['topic']]
            )
        else:
            df = pd.DataFrame(
                [[e['essay'], e['rate']] for e in self.essay_collection.find({'$and': [{'type': test_type}, {'rate': {'$gte': 0}}]})
                    if type(e['essay']) == str and len(e['essay']) > 100 and '강좌 1' not in e['topic']]
            )
        df[1] = df.apply(lambda x: round(x[1]), axis=1)
        df[0] = df.apply(lambda x: preprocess(x[0], mask_number=mask_number, mask_unknown=mask_unknown), axis=1)
        if merge_0_1:
            df.loc[df[1] == 0] = 1

        return df

    def get_spacy_labeled_essays(self, test_type, merge_0_1=True, with_topic=False, mask_number=True, mask_unknown=False):
        if with_topic:
            df = pd.DataFrame(
                [[e['topic'], e['essay'], e['rate']] for e in self.essay_collection.find({'$and': [{'type': test_type}, {'rate': {'$gte': 0}}]})
                    if type(e['essay']) == str and len(e['essay']) > 100 and '강좌' not in e['topic']]
            )
        else:
            df = pd.DataFrame(
                [[e['essay'], e['rate']] for e in self.essay_collection.find({'$and': [{'type': test_type}, {'rate': {'$gte': 0}}]})
                    if type(e['essay']) == str and len(e['essay']) > 100 and '강좌' not in e['topic']]
            )
        def int2text(i):
            if i in [0, 1, 2]:
                return "0"
            elif i in [3, 4]:
                return "1"
            else:
                return "2"
    #            return "{:d}".format(int((i-1)/2))
        rate_loc = 2 if with_topic else 1
        if merge_0_1:
            df[rate_loc] = df.apply(lambda x: int2text(round(x[rate_loc])), axis=1)
            df.loc[df[rate_loc] == "0"] = "1"
        else:
            df[rate_loc] = df.apply(lambda x: str(round(x[rate_loc])), axis=1)
            df.loc[df[rate_loc] == "0"] = "1"
        text_loc = 1 if with_topic else 0
        df[text_loc] = df.apply(lambda x: preprocess(x[text_loc], mask_number=mask_number, mask_unknown=mask_unknown), axis=1)

        return df

    def get_essays_with_keypharse(self, kph):
        return [e for e in self.essay_collection.find({'$and': [{'essay': {'$regex': kph}}, {'rate': {'$gte': 0}}]})
                if type(e['essay']) == str and len(e['essay']) > 100]

    def get_essays_with_topic(self, topic):
        return [e for e in self.essay_collection.find({'$and': [{'topic': {'$regex': topic}}, {'rate': {'$gte': 0}}]})
                if type(e['essay']) == str and len(e['essay']) > 100]

    def export_tsv_labeled_essays(self, test_type, fpath, frac=0.8, random_state=1, merge_0_1=True, with_topic=False):
        try:
            df = self.get_spacy_labeled_essays(test_type, merge_0_1=merge_0_1, with_topic=with_topic). \
                sample(frac=1, random_state=random_state)
            train_end_index = int(df.shape[0]*frac)
            dev_end_index = int((df.shape[0] - train_end_index)/2+train_end_index)
            df[:train_end_index].to_csv(fpath+'/train.tsv', '\t', header=False, index=False)
            df[train_end_index:dev_end_index].to_csv(fpath+'/dev.tsv', '\t', header=False, index=False)
            df[dev_end_index:].to_csv(fpath+'/test.tsv', '\t', header=False, index=False)
        except Exception as e:
            print("Failed to create TSV| %s | %s", fpath, str(e))
        pass

    def export_json_local(self, test_type):
        selected_essay = [e for e in self.essay_collection.find({'$and': [{'type': test_type}, {'rate': {'$gte': 0}}]})
                          if len(e['essay']) > 100]
        return selected_essay