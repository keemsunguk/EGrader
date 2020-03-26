import sys
import os

PROJECT_DIR = '/'.join(os.environ['PWD'].split('/')[:-2])
print(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

from flask import Flask, jsonify, request
from egrader.db_util import DBUtil

app = Flask(__name__)
db_util = DBUtil()

@app.route('/')
@app.route('/index')
def index():
    return jsonify({'message': "Welcome to Keem.Net"})


@app.route('/essay_details')
def essay_details():
    recno = request.args.get('recno', '')
    print(recno)
    tmp = db_util.remote_ec.find_one({'recno': int(recno)})
    if tmp:
        try:
            tmp['_id'] = str(tmp['_id'])
        except ValueError as e:
            tmp = {'message': recno + ' not found '+str(e)}
    else:
        tmp = {'message': recno + ' not found'}
    return jsonify(tmp)
