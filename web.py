from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import numpy as np
from ninda2 import ahpvikor

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/rank', methods=['POST'])
def rank():
    payload = request.get_json()
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
    print(result)
    dict_result = {
      'sorted': result.sorted.to_dict('records'),
      'result': result.result.to_dict('records'),
      'index_sorted': result.index_sorted.tolist(),
      'index_result': result.index_result.tolist(),
      'ahp_result': {
        'cr': result.ahp_result.cr,
        'ci': result.ahp_result.ci,
        'ratio_matrix': result.ahp_result.ratio_matrix.to_dict('records')
      }
      
    }
    return jsonify(dict_result)

@app.route('/ahp', methods=['POST'])
def ahp():
    payload = request.get_json()
    criteria_mat = np.array(payload['criteria_mat'])
    IR = payload['IR']
    n = len(criteria_mat)
    crit_columns = list( f'C{i}' for i in range(1, criteria_mat.shape[0] + 1))
    ahp_result = ahpvikor.ahp(criteria_mat, n, IR, columns=crit_columns)
    print(ahp_result.cr)
    print(ahp_result.ci)
    return jsonify({
      'ratio_matrix': ahp_result.ratio_matrix.to_dict('records'),
      'cr': ahp_result.cr,
      'ci': ahp_result.ci
    })

