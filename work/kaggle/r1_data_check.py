"""Real filtered uniform/long2x order -> R1 seeded order; no training."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from l1_cache import CompactCache
from l2_sampler import load_order
from r1_data import filtered_order
from kaggle_io import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--order',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT.parent/'checks/kaggle-r1-real-order.json');a=p.parse_args()
    cache=CompactCache(a.cache,verify=False);meta=json.loads((a.order/'order.json').read_text())
    original,_=load_order(a.order,cache,meta['file']['sha256'])
    with tempfile.TemporaryDirectory(prefix='r1-real-order-') as tmp:
        actual,new=filtered_order(a.order/'order.npy',cache,meta['file']['sha256'],2026101701,tmp)
        assert np.array_equal(np.sort(actual),np.sort(original))
        assert new['seed']==2026101701 and len(actual)==meta['slots']
        write(a.output,dict(passed=True,rows=len(cache),slots=len(actual),
              source_rule=meta['rule'],source_order_sha256=meta['file']['sha256'],new_order_sha256=new['file']['sha256'],
              seed=new['seed'],membership_unchanged=True,multiplicity_unchanged=True,
              filtered_multiplicity=meta.get('filtered_multiplicity'),
              protected_row_excluded=True,GPU=False,
              limitation='CPU format/membership check only; no evidence of training benefit'))
    print('R1 REAL FILTERED ORDER PASSED; temporary order removed')


if __name__=='__main__':main()
