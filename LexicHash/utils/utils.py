import numpy as np
import os
from os.path import join

def print_banner(x):
    print('\n=================== {} ==================='.format(str(x)))

def print_header(method, fasta_path, n_hash, min_k, max_k, k, **args):
    print('===============================================================================')
    print('=================== SuffixHash Sequence Alignment Estimator ===================')
    print('===============================================================================')
    print(f'running {method} on {fasta_path}')
    print(f'# hash functions: {n_hash}')
    if method=='lexichash':
        print(f'minimum-maximum match length: {min_k}-{max_k}')
    else:
        print(f'k-value: {k}')
        
def print_clocktime(start, end, task):
    print('{} took {} minutes, {} seconds'.format(task, int((end-start)/60), int(end-start)%60))
        
def setup_out_dir(out_dir, args_path,  **args):
    tmp_dir = join(out_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    save_file(args_path, args)
        
def save_file(filepath, x):
    x = np.array(x, dtype=object)
    np.save(filepath, x, allow_pickle=True)
    
def get_run_id(method, n_hash, k=None, **args):
    run_id = f'{"lh" if method=="lexichash" else "mh"}'
    run_id += f'_{n_hash}h'
    if method == 'minhash':
        run_id += f'_{k}k'
    return run_id