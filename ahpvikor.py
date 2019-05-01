import numpy as np
import pandas as pd
from collections import namedtuple

AhpResult = namedtuple('AhpResult', ['prio_weights', 'cr', 'ci'])
AlgoInput = namedtuple('AlgoInput', ['criteria_mat', 'n', 'IR', 'benefit_cols', 'alternatives', 'v', 'id_columns'])

def ahp(criteria_mat, n, IR):
    # Sum by column
    column_sum = criteria_mat.sum(axis=0)

    # Normalize it
    normed = criteria_mat / column_sum

    # Sum the rows
    sum_row = normed.sum(axis=1)

    # compute "priority weights"
    prio_weights = sum_row / 6.0

    # Stack to right
    norm_mat = np.column_stack((normed, sum_row, prio_weights))

    Σ_hk = (criteria_mat * prio_weights).sum(axis=1)
    # print('Hasil kali: ', Σ_hk)
    Σ_hk_prio = Σ_hk / prio_weights
    # print('Weight Hasil kali: ', Σ_hk_prio)

    Σλ = Σ_hk_prio.sum()
    λ_max = Σλ.max() / n
    # print('lambda maks: ', λ_max)
    
    CI = (λ_max - n) / (n - 1)
    CR = CI / IR
    # print('CI=', CI)
    # print('CR=', CR)

    ahp_result = AhpResult(prio_weights=prio_weights, cr=CR, ci=CI)
    
    # print( pd.DataFrame(data=np.column_stack((normed, prio_weights)), columns=(list( f'C{i + 1}' for i in range(n) )) + ['W']))

    return ahp_result
    
def best_worst(mat_alt, benefit_cols):
    # print(f'benefit_cols: {benefit_cols}')
    best_ = []
    worst_ = []
    for idx, ct in enumerate(benefit_cols):
        best = None
        worst = None
        if ct:
            best = mat_alt[:, idx].max()
            worst = mat_alt[:, idx].min()
        else:
            best = mat_alt[:, idx].min()
            worst = mat_alt[:, idx].max()
        best_.append(best)
        worst_.append(worst)
    return np.array(best_), np.array(worst_)

def vikor(alternatives, weights, benefit_cols, v=0.5):
    best, worst = best_worst(alternatives, benefit_cols)
    # print('best: ')
    # print(best)
    # print()

    # print('worst: ')
    # print(worst)
    # print()

    norm = (best - alternatives) / (best - worst)
    # print(norm)
    # exit()

    t1 = norm * weights
    # print(t1)
    # print()
    
    si = t1.sum(axis=1)
    ri = np.max(t1, axis=1)

    si_max = min(si)
    si_min = max(si)
    si_Δ = si_min - si_max

    ri_max = min(ri)
    ri_min = max(ri)
    ri_Δ = ri_min - ri_max

    # print(f"si_max={si_max}")
    # print(f"si_min={si_min}")
    # print(f"si_delta={si_Δ}")

    # print(f"ri_max={ri_max}")
    # print(f"ri_min={ri_min}")
    # print(f"ri_delta={ri_Δ}")
    # print()
    Q = v * ((si - si_max) / si_Δ) + (1 - v) * ((ri - ri_max) / ri_Δ)

    return np.column_stack((t1, si, ri, Q))

def main_algo(_input: AlgoInput):
    ahp_result = ahp(_input.criteria_mat, _input.alternatives.shape[0], _input.IR)
    vikor_result = vikor(_input.alternatives, ahp_result.prio_weights, _input.benefit_cols, _input.v)

    crit_columns = list( f'C{i}' for i in range(1, _input.alternatives.shape[1] + 1))
    vik_columns = crit_columns + [ 'S', 'R', 'Q' ]
    index = _input.id_columns if _input.id_columns is not None else range(1, _input.alternatives.shape[0])

    # print('vikor_result=', vikor_result.shape)
    # print('columns=', len(vik_columns))
    # print('index=', len(index))
    with_q = pd.DataFrame(data=vikor_result, columns=vik_columns, index=index)
    # print(with_q)

    sorted_q = with_q.sort_values(by=['Q'])

    DQ = 1.0 / (_input.alternatives.shape[0] - 1)

    sorted_s = with_q.sort_values(by=['S'])
    sorted_r = with_q.sort_values(by=['R'])

    top_s_index = sorted_s.index[0]
    top_r_index = sorted_r.index[0]
    top_q_index = sorted_q.index[0]

    C1 = (sorted_q['Q'][1] - sorted_q['Q'][0]) > DQ
    C2 = top_q_index == top_r_index == top_s_index

    if C1 and not C2:
        return sorted_q.iloc[0:2]
    if not C1:
        return sorted_q.loc[sorted_q['Q'] < DQ]
    
    return sorted_q.loc[0]

if __name__ == '__main__':
    crit = np.array([
      [1.00, 2.00, 3.00, 5.00, 7.00, 7.00],
      [0.50, 1.00, 3.00, 5.00, 7.00, 7.00],
      [0.33, 0.33, 1.00, 4.00, 5.00, 3.00],
      [0.2, 0.2, 0.25, 1.0, 3.0, 3.0 ],
      [0.14, 0.14, 0.20, 0.33, 1.0, 2.0],
      [0.14, 0.14, 0.33, 0.33, 0.5, 1.0]
    ])
    alternatives = np.array([
      [2.00, 0.00, 4.00, 2.00, 0.00, 4.00],
      [1.00, 0.00, 3.00, 1.00, 0.00, 1.00],
      [4.00, 2.0, 4.0, 0.00, 3.00, 5.00],
      [2.00, 3.0, 4.0, 1.0, 2.0, 3.0 ],
      [4.00, 3.00, 4.00, 2.00, 2.00, 3.00],
      [4.00, 4.00, 4.00, 1.00, 1.00, 4.00]
    ])
    benefit_cols = [ True, False, True, True, True, True ]
    v = 0.5
    IR = 1.24

    index_cols = ['Jl. Amabi', 'Jl. Nangka', 'Jl. Jend. Soeharto', 'Jl. H.R. Koroh', 'Jl. Cak Doko', 'A']

    algo_input = AlgoInput(criteria_mat=crit, 
                      alternatives=alternatives, 
                      v=v, 
                      IR=IR, 
                      benefit_cols=benefit_cols,
                      id_columns=index_cols,
                      n=alternatives.shape[0])

    result = main_algo(algo_input)

    print(result)