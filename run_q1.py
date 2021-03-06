#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')
sys.path.insert(0, 'src/test')

# from etl import get_data
# from analysis import compute_aggregates
from model import train
from craft_attack_patch import load_craft_attack
from train_pgd_orig import train_pgd_attack

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        data = get_data(**data_cfg)

    if 'analysis' in targets:
        with open('config/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)

        # make the data target
        compute_aggregates(data, **analysis_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        # make the data target
        train(data, **model_cfg)
    
    if 'test' in targets:
        with open('config/test-params.json') as fh:
            model_cfg = json.load(fh)
        print('success')
        # make the data target
        load_craft_attack()
        print("done")
        train_pgd_attack()
        print("trained pgd attack")
        # write a successful output
        with open('test/testoutput/test_runresults.txt', 'w') as f:
            f.write('test successful')
    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
