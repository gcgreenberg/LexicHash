import numpy as np
from os.path import exists
from sympy import nextprime
import sys
from multiprocessing import Pool, cpu_count
from time import perf_counter, process_time
import pyfastx

sys.path.append('/home/gcgreen2/alignment/LexicHash')
from LexicHash.utils import seq_utils, utils

BASE_TO_INT = {'A':0, 'T':1, 'G':2, 'C':3, 'a':0, 't':1, 'g':2, 'c':3}
GAP_BOUNDS = [0,*[4**x for x in range(32)]]

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
#     global sketches, sketches_rc, hash_funcs, n_seq, RC
    RC = args['rc']
    start_program = perf_counter()
    
    # OBTAIN READS
    seqs = pyfastx.Fastx(args['fasta_path'])
    
    # MINIMIZERS
    utils.print_banner('COMPUTING MINIMIZERS')
    start = perf_counter()
    hash_func = get_hash_func(**args)
    minimizers, seq_lens, n_seq = get_minimizers(seqs, hash_func, **args)
    utils.print_clocktime(start, perf_counter(), 'computing minimizers')
    
    # PAIRWISE COMPARISON
    utils.print_banner('PERFORMING PAIRWISE COMPARISON')
    start = perf_counter()
    pair_aln_scores = pairwise_comparison(minimizers, n_seq, **args)
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

def get_minimizers(seqs, hash_func, window, n_cpu, rc, **args):
    '''
    Use multiprocessing to compute the sketches for all sequences. 
    
    Returns:
        sketches: Sketches of reads. Each row corresponds to a single read. 
            Each column corresponds to a single hash function
        sketches_rc: Sketches of reverse-complement reads
    '''
    def init_worker_minimizers(hash_func, window, rc):
        global H, RC, W
        H = hash_func
        RC = rc
        W = window
    chunksize = 100
    
    with Pool(processes=n_cpu, initializer=init_worker_minimizers, initargs=(hash_func, window, rc)) as pool:
        all_minimizers = pool.map(get_minimizers_multiproc, seqs, chunksize)
    
    minimizers, minimizers_rc, seq_lens = list(map(list, zip(*all_minimizers))) # list of two-tuples to two lists
    n_seq = len(minimizers)
    if rc: minimizers.extend(minimizers_rc)
    seq_lens = np.array(seq_lens)
    return minimizers, seq_lens, n_seq

    
def get_minimizers_multiproc(seq): 
    '''
    Function called by multiprocessing.
    Computes the sketch of a single sequence.
    
    Args:
        seq: Sequence to sketch
        
    Returns:
        sketch: Sketch of the read. Each entry corresponds to a single mask (i.e. hash function)
        sketch_rc: Sketch of reverse-complement read
    '''
    seq = seq[1] # pyfastx.Fastx iters as (name,seq,comment)
    minimizers = get_seq_minimizers(seq)
    if RC:
        seq_rc = seq_utils.revcomp(seq)
        minimizers_rc = get_seq_minimizers(seq_rc)
    else:
        minimizers_rc = None
    return minimizers, minimizers_rc, len(seq)


def get_seq_minimizers(seq):
    '''
    Function called by multiprocessing.
    Computes the sketch of a single sequence.
    
    Args:
        i: Sequence index
        
    Returns:
        sketch: Sketch of the read. Each entry corresponds to a single hash function
        sketch_rc: Sketch of reverse-complement read
    '''
    hash_vals = H(seq)
    cur_pos = -1
    minimizers = []
    for i in range(len(hash_vals)-W+1):
        pos = i + np.argmin(hash_vals[i:i+W])
        if pos != cur_pos:
            cur_pos = pos
            minimizers.append((hash_vals[cur_pos], cur_pos))
    return minimizers


def get_hash_func(hash_type, **args): 
    '''
    Create randomized hash functions.
    
    Args:
        n_bits: Number of bits for hash-values
        k: Length of substrings to hash
        n_hash: Number of hash functions to create.
        
    Returns:
        hash_funcs: Hash functions (see class random_hash_func)
    '''
    if hash_type == 'lexichash':
        return lexic_hash_func(args['k'])
    elif hash_type == 'random':
        max_coef = 2**args['n_bits']
        max_hash = nextprime(max_coef)
        return random_hash_func(max_coef, max_hash, args['k'])
    else:
        raise Exception('hash_type options are lexichash or random')
 
    
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
    
    
class lexic_hash_func():
    def __init__(self, k):
        self.mask = np.random.randint(4, size=k)
        self.k = k
        
    def hash_val(self, kmer):
        val = 0
        for b,bm in zip(kmer,self.mask):
            val <<= 2
            val += b ^ bm
        return val
    
    def __call__(self, seq):
        n_kmers = len(seq)-self.k+1; assert n_kmers>0
        seq_int_repr = np.array([BASE_TO_INT[b] for b in seq])
        hash_vals = [self.hash_val(seq_int_repr[i:i+self.k]) for i in range(n_kmers)]
        return hash_vals 
    
    
# class Minimizers():
#     def __init__(self):
        
    

#################################################
#            PAIRWISE COMPARISON                #
#################################################

def pairwise_comparison(minimizers, n_seq, max_errors, **args):
    """
    Perform the pairwise comparison component of the MinHash pipeline.

    Args:
        n_hash: Number of hash functions used.
        min_n_col: Minimum number of minhash collisions to consider between a pair of sequence sketches.
        
    Returns:
        pair_aln_scores: Dict of pair:similarity score. Pair is a tuple of the form (id1,id2,+/-).
    """
    all_matching_sets = hash_table(minimizers)
    all_hits = collect_hits(all_matching_sets, n_seq, max_errors)
    pair_aln_scores = get_pair_aln_scores(all_hits)
    return pair_aln_scores


def hash_table(minimizers):
    '''
    Function called by multiprocessing.
    Computes the pairwise comparison using a hash table.
    Partitions the set of hash-values for a single hash function into
        corresponding list of sets of sequence indices.
    
    Args:
        sketch_idx: Index in sketch to consider (i.e. hash function index)
        
    Returns:
        matching_sets: Dict of hash-value:list of sets of sequence indices with same hash-value. 
    '''
    matching_sets = {}
    for i,seq_minimizers in enumerate(minimizers):
        for val,pos in seq_minimizers:
            if val in matching_sets:
                matching_sets[val].add((i,pos))
            else:
                matching_sets[val] = {(i,pos)}
    return matching_sets 


def collect_hits(all_matching_sets, n_seq, max_errors, min_score=0, score_decay=0.5):
    def gap_score(val1, val2):
        gap = val1 ^ val2
        for i in range(len(GAP_BOUNDS)-1):
            if gap>=GAP_BOUNDS[i] and gap<GAP_BOUNDS[i+1]:
                return score_decay**(i)
    
    def get_hash_pairs():
        hash_pairs = {}
        sorted_hash_vals = np.sort(list(all_matching_sets))
        for i,val in enumerate(sorted_hash_vals):
            hash_pairs[val] = set()
            j = i
            while j<len(sorted_hash_vals) and (val ^ sorted_hash_vals[j])<=4**max_errors:
                hash_pairs[val].add(sorted_hash_vals[j])
                j += 1
        return hash_pairs
           
    def collect_hits_for_hash_pair(val1, val2, all_hits):
        seq_minimizers = all_matching_sets[val1].union(all_matching_sets[val2])
        for i1,pos1 in seq_minimizers:
            for i2,pos2 in seq_minimizers:
                _i1,_i2 = i1%n_seq, i2%n_seq
                is_rc = '-' if (i1>=n_seq and i2<n_seq) or (i1<n_seq and i2>=n_seq) else '+' 
                key = (_i1,_i2,is_rc)
                if _i1<_i2 and (i1<n_seq or i2<n_seq):
                    score = gap_score(val1,val2)
                    if key in all_hits:
                        all_hits[key].append((pos1-pos2,pos1,pos2,score))
                    else:
                        all_hits[key] = [(pos1-pos2,pos1,pos2,score)]
          
    if max_errors is None: max_errors = 0 
    hash_pairs = get_hash_pairs()
    all_hits, seq_pair_score = {}, {}
    for val1 in hash_pairs:
        for val2 in hash_pairs[val1]:
            collect_hits_for_hash_pair(val1, val2, all_hits)
#     remove_low_score_pairs(all_hits, seq_pair_score)
    return all_hits

            
def get_pair_aln_scores(all_hits):
    ''' 
    Processes the output of the multiprocessing step.
    
    Args:
        all_matching_sets: Iterator of dicts of form similarity score:list of sets of sequence indices
        n_hash: Number of hash functions.
        min_n_col: Minimum number of minhash collisions to consider between a pair of sequence sketches.
        
    '''
    pair_aln_scores = {}
    for key,hits in all_hits.items():
        score = get_chaining_score(hits)
        pair_aln_scores[key] = score 
                        
    return pair_aln_scores
                    

def get_chaining_score(hits, eps=20):
    hits = np.array(hits)
    hits = hits[hits[:,1].argsort()] # sort on pos1
    hits = hits[hits[:,0].argsort(kind='mergesort')] # sort on pos1-pos2; mergesort keeps the first sort in place

    p, max_score, cur_score = 0, 0, 0
    for i in range(len(hits)):
        cur_score += hits[i][3]
        if i==len(hits)-1 or hits[i,0]-hits[i+1,0] > eps:
            if cur_score > max_score:
                max_score = cur_score
    return max_score

    
    
    
    
    