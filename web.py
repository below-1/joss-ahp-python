from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import numpy as np
from ninda2 import ahpvikor

app = Flask(__name__)
CORS(app)

@app.route('/rank', methods=['POST'])
def rank():
    payload = request.get_json()
    print(payload)
    alternatives = np.array(payload['alternatives'])
    indices = np.array(payload['indices'])
    benefit_cols = [ bool(x) for x in payload['benefit_cols'] ]
    v = payload['v']
    IR = payload['IR']
    criteria_mat = np.array(payload['criteria_mat'])
    algo_input = ahpvikor.AlgoInput(criteria_mat=criteria_mat, 
                      alternatives=alternatives, 
                      v=v,
                      IR=IR, 
                      benefit_cols=benefit_cols,
                      id_columns=indices,
                      n=alternatives.shape[0])
    result = ahpvikor.main_algo(algo_input)
    print('result=', result)
    return result.to_json(orient='table')

