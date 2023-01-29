#!/home/gcgreen2/envs/py3.9venv/bin/python3
import sys
import os
import argparse
from os.path import join

from LexicHash import LexicHash, MinHash
from LexicHash.utils import utils
  
def check_args(args):
    assert args['method'] in ['lexichash', 'minhash'], 'method argument must be either "lexichash" or "minhash"'
    if args['method'] == 'minhash':
        assert args['k'] is not None, 'must set a k-value for minhash'
        assert args['k'] >= 7 and args['k'] <= 24, 'k-value must be between 7 and 24'
    assert args['n_cpu'] <= os.cpu_count(), f'only {os.cpu_count()} cpus available'
    

def add_args(args):
    run_id = utils.get_run_id(**args)
    args['n_bits'] = 30
    args['aln_path'] = join(args['out_dir'], f'aln_{run_id}.tsv')
    args['sketch_path'] = join(args['out_dir'], 'tmp', f'sketches_{run_id}.npy')
    args['args_path'] = join(args['out_dir'], 'tmp', f'args_{run_id}.npy')
    return args

def parse_args():
    parser = argparse.ArgumentParser(description='Sequence Similarity Estimation via Lexicographic Comparison of Hashes')
    parser.add_argument('--out', type=str, dest='out_dir', help='output directory', required=True)
    parser.add_argument('--fasta', type=str, dest='fasta_path', help='path to reads fasta', required=True)
    parser.add_argument('--method', type=str, help='alignment method (lexichash/minhash)', default='lexichash')
    parser.add_argument('--n_hash', type=int, default=500, help='number of hash functions to use')
    parser.add_argument('--k', type=int, help='k-value (for minhash only)', default=16)
    parser.add_argument('--min_k', type=int, help='min match length (for lexichash only)', default=14)
    parser.add_argument('--max_k', type=int, help='max match length (for lexichash only)', default=32)
    parser.add_argument('--no_rc', action='store_false', help='do not account for reverse-complements', dest='rc')
    parser.add_argument('--min_n_col', type=int, help='min # of minhash collisions (for minhash only)', default=1)
    parser.add_argument('--n_cpu', type=int, help='number of cpus to use', default=os.cpu_count())
    parser.add_argument('--alpha', type=float, help='alpha*n_seq pairs will be output', default=None)
    parser.set_defaults(rc=True)
    return vars(parser.parse_args())

def main():
    args = parse_args()
    args = add_args(args)
    check_args(args)
    utils.setup_out_dir(**args)
    utils.print_header(**args)
    
    if args['method'] == 'lexichash':
        LexicHash.find_overlaps(**args)
    elif args['method'] == 'minhash':
        MinHash.find_overlaps(**args)   

if __name__ == '__main__':
    main()
