import numpy as np
from collections import namedtuple
from ninda2 import ahp
from ninda2 import vikor

AlgoInput = namedtuple('AlgoInput', ['criteria_mat', 'n', 'IR', 'benefit_cols', 'alternatives', 'v'])

def main_algo(_input: AlgoInput):
    norm_mat = ahp.normalize(_input.criteria_mat, _input.n, _input.IR)
    
    pass