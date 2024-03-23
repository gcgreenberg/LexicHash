import numpy as np
from os.path import exists
from sympy import nextprime
import sys
from multiprocessing import Pool, cpu_count
from time import perf_counter, process_time
import pyfastx
from LexicHash.utils import seq_utils, utils


def find_overlaps(**args):
    """
    Runs the MinHash pipeline on a set of long-read sequencing reads
    to find similarities between pairs of reads.

    Args (in args dictionary):
        fasta_path: Path of the reads in fasta format
        out: Path of the output directory to write alignment file
        n_hash: Number of hash functions to consider (int)
        k: k-value for the MinHash pipeline (int)
        rc: Whether or not to consider reverse-complement reads (boolean)
        min_n_col: Minimum number of minhash collisions to consider for output
        
    Globals:
        sketches: Sketches of reads. Each row corresponds to a single read. 
            Each column corresponds to a single mask (i.e. hash function)
        sketches_rc: Sketches of reverse-complement reads
        hash_funcs: Each hash function is used to find a minhash value.
        n_seq: Number of sequencing reads
        RC: Whether to consider reverse-complement reads in whole pipeline (typically True)
    """
    # global sketches, sketches_rc, hash_funcs, n_seq, RC
    # RC = args['rc']
    start_program = perf_counter()
    
    # OBTAIN READS
    seqs = pyfastx.Fastx(args['fasta_path'])
    
    # SKETCHING
    utils.print_banner('SKETCHING READS')
    start = perf_counter()
    hash_funcs = get_hash_funcs(**args)
    sketches, sketches_rc, seq_lens = sketching(seqs, hash_funcs, **args)
    n_seq = len(sketches)
    utils.print_clocktime(start, perf_counter(), 'sketching')
    
    # PAIRWISE COMPARISON
    utils.print_banner('PERFORMING PAIRWISE COMPARISON')
    start = perf_counter()
    pair_aln_scores = pairwise_comparison(sketches, seq_lens, n_seq, **args)
    utils.print_clocktime(start, perf_counter(), 'pairwise comparison')
    
    # WRITE RESULTS
    write_overlaps(pair_aln_scores, **args)
    utils.print_clocktime(start_program, perf_counter(), 'full process')
    
    
def write_overlaps(pair_aln_scores, aln_path, **args):
    '''
    Writes the overlap pairs to output file. 
    
    Args:
        pair_aln_scores: Dict of pair:alignment score. Pair is a tuple of the form (id1,id2,+/-).
        aln_path: Output tsv filepath. Each row is a pair with columns (id1,id2,+/-,match-length)
    '''
    print(f'# overlaps: {len(pair_aln_scores)}')
    with open(aln_path, 'w') as fh:
        for pair,score in pair_aln_scores.items():
            id1,id2,is_rc = pair
            line = '\t'.join([str(id1), str(id2), str(round(score,1)), is_rc])
            fh.write(line+'\n')
                
                
#################################################
#                SKETCHING                      #
#################################################

def sketching(seqs, hash_funcs, n_hash, k, n_cpu, sketch_path=None, **args):
    '''
    Use multiprocessing to compute the sketches for all sequences. 
    
    Returns:
        sketches: Sketches of reads. Each row corresponds to a single read. 
            Each column corresponds to a single hash function
        sketches_rc: Sketches of reverse-complement reads
    '''
    chunksize = 100
    with Pool(processes=n_cpu, initializer=init_worker_sketching, initargs=(hash_funcs, rc, k)) as pool:
        all_sketches = pool.map(get_seq_sketch, seqs, chunksize)
    
    sketches, sketches_rc, seq_lens = list(map(list, zip(*all_sketches))) # list of two-tuples to two lists
    sketches, sketches_rc = np.array(sketches), np.array(sketches_rc) if RC else None
    seq_lens = np.array(seq_lens)
    return sketches, sketches_rc, seq_lens

def init_worker_sketching(hash_funcs, rc, k):
    global shared_hash_funcs, RC, K
    shared_hash_funcs = hash_funcs
    RC = rc
    K = k

def get_seq_sketch(seq):
    '''
    Function called by multiprocessing.
    Computes the sketch of a single sequence.
    
    Args:
        i: Sequence index
        
    Returns:
        sketch: Sketch of the read. Each entry corresponds to a single hash function
        sketch_rc: Sketch of reverse-complement read
    '''
    seq = seq[1] # pyfastx.Fastx sequence is tuple (name,seq,comment)
    sketch = [min(h(seq)) for h in hash_funcs]
    if RC:
        seq_rc = seq_utils.revcomp(seq)
        sketch_rc = [min(h(seq_rc)) for h in hash_funcs]
        return sketch, sketch_rc, len(seq)
    return sketch, None, len(seq)
        


def get_hash_funcs(n_bits, k, n_hash, **args): 
    '''
    Create randomized hash functions.
    
    Args:
        n_bits: Number of bits for hash-values
        k: Length of substrings to hash
        n_hash: Number of hash functions to create.
        
    Returns:
        hash_funcs: Hash functions (see class random_hash_func)
    '''
    max_coef = 2**n_bits
    max_hash = nextprime(max_coef)
    hash_funcs = [random_hash_func(max_coef, max_hash, k) for _ in range(n_hash)]
    return hash_funcs
 
    
class random_hash_func():
    '''
    Random hash function. Hashes k-mer substrings of a sequence.
    Function consists of coeficients a and b, maximum hash-value (highest prime above 2**n_bits).
    Hash-value uses Python's built-in `hash` function.
    Hash-value is (a*hash(k-mer) + b) % max_hash.
    '''
    def __init__(self,max_coef,max_hash,k):
        self.a = np.random.randint(max_coef)
        self.b = np.random.randint(max_coef)
        self.max_hash = max_hash
        self.k = k
    
    def __call__(self, seq):
        n_kmers = len(seq)-self.k+1; assert n_kmers>0
        hashes = np.array([hash(seq[i:i+self.k]) for i in range(n_kmers)])
        hashes = (hashes*self.a + self.b) % self.max_hash
        return hashes
    

#################################################
#            PAIRWISE COMPARISON                #
#################################################

def pairwise_comparison(sketches, seq_lens, n_seq, n_hash, k, min_n_col, n_cpu, **args):
    """
    Perform the pairwise comparison component of the MinHash pipeline.

    Args:
        n_hash: Number of hash functions used.
        min_n_col: Minimum number of minhash collisions to consider between a pair of sequence sketches.
        
    Returns:
        pair_aln_scores: Dict of pair:similarity score. Pair is a tuple of the form (id1,id2,+/-).
    """
    all_matching_sets = hash_table_multiproc(sketches, k, n_hash)
    pair_aln_scores = process_matching_sets(all_matching_sets, seq_lens, n_hash, min_n_col)
    return pair_aln_scores


def hash_table_multiproc(sketches, k, n_hash):
    args = (i for i in range(n_hash))
    chunksize = int(np.ceil(n_hash/cpu_count()/4))
    with Pool(processes=n_cpu, initializer=init_worker_hash_table, initargs=(sketches,k)) as pool:
        all_matching_sets = pool.map(get_matching_sets, args, chunksize)
    all_matching_sets = np.concatenate(all_matching_sets)
    return all_matching_sets    

def init_worker_hash_table(sketches,k):
    global shared_sketches, K
    shared_sketches = sketches
    K = k

def get_matching_sets(sketch_idx):
    '''
    Function called by multiprocessing.
    Computes the pairwise comparison using a hash table.
    Partitions the set of minhash-values for a single hash function into
        corresponding list of sets of sequence indices.
    
    Args:
        sketch_idx: Index in sketch to consider (i.e. hash function index)
        
    Returns:
        matching_sets: Dict of similarity score:list of sets of sequence indices with same minhash-value. 
    '''
    matching_sets = {}
    for i in range(2*n_seq if RC else n_seq):
        val = sketches[i,sketch_idx] if i<n_seq else sketches_rc[i%n_seq,sketch_idx]
        if val in matching_sets:
            matching_sets[val].add(i)
        else:
            matching_sets[val] = {i}
    matching_sets = [s for s in matching_sets.values() if len(s)>1]
    return matching_sets 


def process_matching_sets(all_matching_sets, seq_lens, n_hash, min_n_col):
    ''' 
    Processes the output of the multiprocessing step.
    
    Args:
        all_matching_sets: Iterator of dicts of form similarity score:list of sets of sequence indices
        n_hash: Number of hash functions.
        min_n_col: Minimum number of minhash collisions to consider between a pair of sequence sketches.
        
    '''
    def index_matching_sets():
        seq_idxs = range(2*n_seq if RC else n_seq) # will either be {1,...,n_seq} or {1,...,2*n_seq}
        seq_to_set_idxs = {i:set() for i in seq_idxs}
        for j,s in enumerate(all_matching_sets):
            for i in s:
                seq_to_set_idxs[i].add(j)
        return seq_to_set_idxs
    
    def get_pair_aln_scores(seq_to_set_idxs):
        pair_aln_scores = {}
        for cur_set in all_matching_sets:
            for i1 in cur_set: # seq index 1
                for i2 in cur_set: # seq index 2
                    _i1,_i2 = i1%n_seq, i2%n_seq
                    if _i1<_i2 and (i1<n_seq or i2<n_seq):
                        is_rc = '-' if (i1>=n_seq and i2<n_seq) or (i1<n_seq and i2>=n_seq) else '+' 
                        if (_i1,_i2,is_rc) not in pair_aln_scores:
                            n_col = len(seq_to_set_idxs[i1].intersection(seq_to_set_idxs[i2]))
                            if n_col > min_n_col:
                                score = (seq_lens[_i1]+seq_lens[_i2]) * n_col/(n_hash+n_col)
                                pair_aln_scores[(_i1,_i2,is_rc)] = score 
        return pair_aln_scores
                                    
    
    seq_to_set_idxs = index_matching_sets()
    pair_aln_scores = get_pair_aln_scores(seq_to_set_idxs)
    return pair_aln_scores
