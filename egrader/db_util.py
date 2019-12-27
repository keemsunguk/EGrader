from pymongo import MongoClient
import pandas as pd
from egrader.config import config

conf = config.get_config()
REMOTE_MONGO = conf['RemoteMongo']
LOCAL_MONGO = conf['LocalMongo']


class DBUtil:
    def __init__(self):
        remote_client = MongoClient(REMOTE_MONGO)
        local_client = MongoClient(LOCAL_MONGO)
        self.redb = remote_client['nlp']
        self.ledb = local_client['essay']
        self.local_ec = self.ledb['essay']
        self.remote_ec = self.redb['essays']

    def describe_db(self):
        try:
            print('==== LOCAL ====')
            print(self.ledb.list_collection_names())
            print('Total Essay:', self.local_ec.count_documents({}))
            print('Total SAT:', self.local_ec.count_documents({'$and': [{'type': 'SAT'}, {'rate': {'$gte': 0}}]}))
            print('Total TOEFL:', self.local_ec.count_documents({'$and': [{'type': 'TOEFL'}, {'rate': {'$gt': 0}}]}))
            print('Total GRE:', self.local_ec.count_documents({'$and': [{'type': 'GRE'}, {'rate': {'$gte': 0}}]}))
        except Exception as e:
            print("No local mongo connection found: %s", str(e))

        try:
            print('==== REMOTE ====')
            print(self.redb.list_collection_names())
            print('Total Essay:', self.remote_ec.count_documents({}))
            print('Total SAT:', self.remote_ec.count_documents({'$and': [{'type': 'SAT'}, {'rate': {'$gte': 0}}]}))
            print('Total TOEFL:', self.remote_ec.count_documents({'$and': [{'type': 'TOEFL'}, {'rate': {'$gt': 0}}]}))
            print('Total GRE:', self.remote_ec.count_documents({'$and': [{'type': 'GRE'}, {'rate': {'$gte': 0}}]}))
        except Exception as e:
            print("No remote mongo connection found: %s", str(e))

    def get_sat_essays(self):
        return [e for e in self.remote_ec.find({'$and': [{'type': 'SAT'}, {'rate': {'$gte': 0}}]})
                if len(e['essay']) > 100]

    def get_toefl_essays(self):
        return [e for e in self.remote_ec.find({'$and': [{'type': 'TOEFL'}, {'rate': {'$gt': 0}}]})
                if len(e['essay']) > 100]

    def get_gre_essays(self):
        return [e for e in self.remote_ec.find({'$and': [{'type': 'GRE'}, {'rate': {'$gte': 0}}]})
                if type(e['essay']) == str and len(e['essay']) > 100]

    def get_labeled_essays(self, test_type, merge_0_1=True):
        df = pd.DataFrame(
            [[e['essay'], e['rate']] for e in self.remote_ec.find({'$and': [{'type': test_type}, {'rate': {'$gte': 0}}]})
                if type(e['essay']) == str and len(e['essay']) > 100]
        )
        df[1] = df.apply(lambda x: round(x[1]), axis=1)
        if merge_0_1:
            df.loc[df[1] == 0] = 1

        return df

    def get_spacy_labeled_essays(self, test_type, merge_0_1=True):
        df = pd.DataFrame(
            [[e['essay'], e['rate']] for e in self.remote_ec.find({'$and': [{'type': test_type}, {'rate': {'$gte': 0}}]})
                if type(e['essay']) == str and len(e['essay']) > 100]
        )

        def int2text(i):
            if i in [0, 1, 2]:
                return "0"
            elif i in [3, 4]:
                return "1"
            else:
                return "2"
    #            return "{:d}".format(int((i-1)/2))

        df[1] = df.apply(lambda x: int2text(round(x[1])), axis=1)
        if merge_0_1:
            df.loc[df[1] == 0] = 1

        return df
