"""L4 configuration of the reviewed L3 assembly; local build only."""
import argparse
from build_l3 import build

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gpu-smoke',action='store_true')
    p.add_argument('--cache-source',default='thisarivson/nlp-hw02-compact-cache-verified/2')
    a=p.parse_args()
    target,_=build(a.gpu_smoke,a.cache_source,continuation='l4')
    print(target)
