import numpy as np
import sys
from os.path import exists
from multiprocessing import Pool, cpu_count
from time import perf_counter, process_time
import pyfastx
from LexicHash.utils import utils, seq_utils

BASE_TO_INT = {'A':0, 'T':1, 'G':2, 'C':3, 'a':0, 't':1, 'g':2, 'c':3}

def find_overlaps(**args):
    """
    Runs the LexicHash pipeline on a set of long-read sequencing reads
    to find similarities between pairs of reads.

    Args (in args dictionary):
        fasta_path: Path of the reads in fasta format
        out: Path of the output directory to write alignment file
        n_hash: Number of hash functions to consider (int)
        min_k: Minimum match length to consider for output (int)
        max_k: Maximum match length to consider for output 
        rc: Whether or not to consider reverse-complement reads (boolean)
        alpha: Outputs alpha*n_seq total pairs
        
    Globals:
        sketches: Sketches of reads. Each row corresponds to a single read. 
            Each column corresponds to a single mask (i.e. hash function)
        masks: Each mask indicates lexicographic order for sketching procedure.
            `masks` has shape (n_hash, MAX_K)
        MIN_K: Minimum match-length to consider for output
        MAX_K: Maximum match-length to consider for output
        RC: Whether to consider reverse-complement reads in whole pipeline (typically True)
    """
    global sketches, masks, MIN_K, MAX_K, RC # global variables needed for multiprocessing
    MIN_K = args['min_k']
    MAX_K = args['max_k']
    RC = args['rc']
    start_program = perf_counter()
    
    # OBTAIN READS (no RAM)
    seqs = pyfastx.Fastx(args['fasta_path'])
    
    # SKETCHING
    utils.print_banner('SKETCHING READS')
    start = perf_counter()
    masks = get_masks(args['n_hash'])
    sketches, n_seq = sketching(seqs, **args)
    print(f'# sequences: {n_seq}')
    utils.print_clocktime(start, perf_counter(), 'sketching')

    # PAIRWISE COMPARISON
    utils.print_banner('PERFORMING PAIRWISE COMPARISON')
    start = perf_counter()
    pair_match_lens = pairwise_comparison(n_seq, **args)
    utils.print_clocktime(start, perf_counter(), 'pairwise comparison')
    
    # WRITE RESULTS
    write_overlaps(pair_match_lens, **args)
    utils.print_clocktime(start_program, perf_counter(), 'full process')
    
    
def write_overlaps(pair_match_lens, aln_path, **args):
    '''
    Writes the overlap pairs to output file. 
    
    Args:
        pair_match_lens: Dict of pair:match-length. Pair is a tuple of the form (id1,id2,+/-).
        aln_path: Output tsv filepath. Each row is a pair with columns (id1,id2,+/-,match-length)
    '''
    print(f'# overlaps: {len(pair_match_lens)}')
    with open(aln_path, 'w') as fh:
        for pair,match_len in pair_match_lens.items():
            id1,id2,is_rc = pair
            line = '\t'.join([str(id1), str(id2), str(match_len), is_rc])
            fh.write(line+'\n')
                
#################################################
#                SKETCHING                      #
#################################################

def sketching(seqs, n_hash, n_cpu, **args):
    '''
    Use multiprocessing to compute the sketches for all sequences. 
    
    Returns:
        sketches: Sketches of reads. Each row corresponds to a single read. 
            Each column corresponds to a single mask (i.e. hash function).
            Includes reverse-complements unless no_rc is set
        n_seq: Number of sequences
    '''
    chunksize = 100
    with Pool(processes=n_cpu) as pool:
        all_sketches = pool.map(get_seq_sketch, seqs, chunksize)
    sketches, sketches_rc = list(map(list, zip(*all_sketches))) # list of two-tuples to two lists
    n_seq = len(sketches)
    sketches = np.concatenate((sketches, sketches_rc)) if RC else np.array(sketches)
    return sketches, n_seq


def get_seq_sketch(seq): 
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
    seq_int_repr = np.array([BASE_TO_INT[b] for b in seq])
    sketch = [lexicographic_first(seq_int_repr, mask) for mask in masks]
    if RC:
        seq_rc = seq_utils.revcomp(seq)
        seq_int_repr = np.array([BASE_TO_INT[b] for b in seq_rc])
        sketch_rc = [lexicographic_first(seq_int_repr, mask) for mask in masks]
    else:
        sketch_rc = None
    return sketch, sketch_rc
    
    
def lexicographic_first(seq,mask):
    '''
    Finds the lexicographic first substring of input sequence given 
        lexicographic order indicated by the input mask. 
    First finds candidate locations in the sequence which correspond to
        the starting locations of the longest substrings exactly matching 
        the prefix of the mask.
    Next extends these candidate substrings to find the lexicograhpic first
        one according the mask.
    
    Args:
        seq: Sequence to consider
        mask: Lexicographic order to consider
        
    Returns:
        min-hash: Rank (hash value) of lexicographic first substring.
    '''
    def get_candidate_locs():
        j = 0 
        cur_idxs = np.arange(len(seq)-MIN_K)
        while j < MAX_K:
            next_idxs = np.where(seq[cur_idxs] == mask[j])[0]
            if len(next_idxs)==0:
                return cur_idxs-j, j
            elif len(next_idxs)==1:
                return [cur_idxs[next_idxs[0]]-j], j+1
            cur_idxs = cur_idxs[next_idxs]+1
            if cur_idxs[-1] == len(seq): 
                cur_idxs = cur_idxs[:-1]  
            j += 1
        return cur_idxs-MAX_K, MAX_K
    
    def extend_candidates(candidate_locs, n_matching):
        j = n_matching
        while len(candidate_locs) > 1 and j < MAX_K:
            best = 4 # will always be overwritten immediately
            next_candidates = []
            for loc in candidate_locs:
                if loc+j < len(seq):
                    bm = (seq[loc+j]-mask[j]) % 4
                    if bm < best:
                        best = bm
                        next_candidates = [loc]
                    elif bm == best:
                        next_candidates.append(loc)
            if len(next_candidates) == 0: break
            candidate_locs = next_candidates
            j += 1    
        return candidate_locs[0]
    
    def hash_val(lex_first_idx):
        val = 0
        for b in seq[lex_first_idx:lex_first_idx+MAX_K]:
            val <<= 2
            val += b
        base_extend = max(0, MAX_K-(len(seq)-lex_first_idx))
        if base_extend > 0: # used if substring is at end of sequence (corner case)
            for b in mask[-base_extend:]:
                val <<= 2 # 2 bits per base
                val += b ^ 3 # extend with lexicographically last rank
        return val

    candidate_locs, n_matching = get_candidate_locs() # n_matching is length exactly matching the mask
    lex_first_idx = extend_candidates(candidate_locs, n_matching)
    return hash_val(lex_first_idx)


def get_masks(n_masks): 
    '''
    Create randomized masks.
    
    Args:
        n_masks: Number of masks to create. Corresponds to input argument n_hash.
    Returns:
        masks: Numpy array (n_masks, max match length). Entries are integers in {0,1,2,3}.
    '''
    mask_len = MAX_K
    masks = np.random.randint(4, size=(n_masks,mask_len))
    return masks


#################################################
#            PAIRWISE COMPARISON                #
#################################################

def pairwise_comparison(n_seq, n_hash, n_cpu, alpha, **args): 
    """
    Perform the pairwise comparison component of the LexicHash pipeline.

    Args:
        n_seq: Number of sequences
        n_hash: Number of hash functions used.
        n_cpu: Number of cpus to use
        alpha: Will aggregate alpha*n_seq pairs
        
    Returns:
        pair_match_lens: Dict of pair:match-length. Pair is a tuple of the form (id1,id2,+/-).
    """
    n_pairs = n_seq * alpha if alpha is not None else np.inf
    all_matching_sets = prefix_tree_multiproc(n_hash, n_cpu)
#     min_k = get_k_thresh(all_matching_sets) if min_k is None else min_k
    pair_match_lens = bottom_up_pair_aggregation(all_matching_sets, n_seq, n_pairs)
    return pair_match_lens


def prefix_tree_multiproc(n_hash, n_cpu):
    '''
    Use multiprocessing to compute the pairwise comparisons. 
    Each process considers a single hash function.
    
    Returns:
        sketches: Sketches of reads. Each row corresponds to a single read. 
            Each column corresponds to a single mask (i.e. hash function)
        sketches_rc: Sketches of reverse-complement reads
    '''
    args = (i for i in range(n_hash))
    chunksize = int(np.ceil(n_hash/n_cpu/4))
    with Pool(processes=n_cpu) as pool:
        all_matching_sets = pool.map(get_matching_sets, args, chunksize)
    return all_matching_sets


def get_matching_sets(sketch_idx):
    '''
    Function called by multiprocessing.
    Efficiently computes the pairwise comparison using a structure similar to a prefix-tree.
    Starts at root of the prefix tree, which is represented as the set of all sequence indices.
    Next, partition the root set based on the groups of substrings with matching first base.
    Continue partitioning until a depth of MAX_K or until no non-singleton sets remain.
    The partition at depth k is a list of sets, where the number of sets corresponds to the number of
        subtrees at depth k in the prefix tree, and each set corresponds to the sequence indices 
        in that subtree.
    
    Args:
        sketch_idx: Index in sketch to consider (i.e. hash function index)
        
    Returns:
        matching_sets: Dict of match-length: list of sets of sequence indices
            with match of corresponding length 
    '''
    subtrees = [list(range(len(sketches)))]
    matching_sets = {}
    for k in range(MAX_K): # current position in all substrings k (i.e. match-length - 1)
        next_subtrees = []
        for seq_idxs in subtrees: 
            partition = {0:[],1:[],2:[],3:[]}
            chars = (sketches[seq_idxs, sketch_idx] >> 2*(MAX_K-k-1)) & 3
            for char,seq_idx in zip(chars,seq_idxs):
                partition[char].append(seq_idx)
            partition = [p for p in partition.values() if len(p)>1]
            next_subtrees.extend(partition)
        if len(next_subtrees) == 0:
            return matching_sets
        subtrees = next_subtrees
        if k+1 >= MIN_K: # k+1 because a match at index 0 is a length-1 match, e.g.
            matching_sets[k+1] = subtrees.copy()
    return matching_sets
                   

def bottom_up_pair_aggregation(all_matching_sets, n_seq, n_pairs):
    '''
    Aggregate pairs of similar sequences from the ``bottom" up. Starts with maximum match-length (MAX_K),
        and for each set in the list of sets at that level, it adds all possible pairs of sequence indices
        to the dictionary.
        
    Args:
        matching_sets_comb: Single giant dict of match-length:list of all sets of sequence indices
    
    Returns:
        pair_match_lens: Dict of pairs:match-length. Pair is a tuple of the form (id1,id2,+/-).
    '''
# go from combined dict to dict of seq idx pairs (i1,i2) --> (match_len, +/-)
    pair_match_lens = {}
    for k in range(MAX_K, MIN_K-1, -1): # start from bottom to give max match length to each pair
        for matching_sets in all_matching_sets:
            for matching_set in matching_sets.get(k, []):
                for i1 in matching_set: # seq index 1
                    for i2 in matching_set: # seq index 2
                        _i1,_i2 = i1%n_seq, i2%n_seq
                        if _i2 > _i1:
                            is_rc = '-' if (i1>=n_seq and i2<n_seq) or (i1<n_seq and i2>=n_seq) else '+'
                            key = (_i1,_i2,is_rc)
                            if key not in pair_match_lens:
                                pair_match_lens[key] = k
                                if len(pair_match_lens) == n_pairs:
                                    return pair_match_lens
    return pair_match_lens




# TO USE, YOU MUST CHANGE bottom_up_pair_aggregation SO THAT YOU CAN SET THE MIN_K, THEN GET 1 FULL SAMPLE TREE, THEN CHANGE THE get_k_thresh ACCORDINGLY 
def get_k_thresh(all_matching_sets, max_quantile=0.99):
    '''
    This function finds a minimum match-length to consider based on the output of the prefix-tree step.
    Calculates the match-length PMF by determining the number of pairs at each length. 
    The minimum match-length is set to the inflection point of the log-PMF. 
    This method has proven to be reliable.
    '''
    def get_all_n_pairs_arr():
        k_vals = np.arange(MAX_K+1)
        all_n_pairs_arr = np.zeros(MAX_K+1)
        for matching_sets in all_matching_sets:
            cur_n_pairs_arr = get_n_pairs_arr(matching_sets)
            for k,n_pairs in zip(k_vals,cur_n_pairs_arr):
                all_n_pairs_arr[k] += n_pairs
        return k_vals, all_n_pairs_arr
    
    def get_n_pairs_arr(matching_sets):
        n_pairs_arr = np.zeros(MAX_K+1)
        for k in np.sort(list(matching_sets))[::-1]:
            n_pairs = 0
            for s in matching_sets[k]:
                n_pairs += len(s)*(len(s)-1)//2
            n_pairs_arr[k] = n_pairs
            if k <= MAX_K: 
                n_pairs_arr[k] -= sum(n_pairs_arr[k+1:])
        return n_pairs_arr
  
    def get_extreme_value_cdf(n_pairs_arr):
        n_hash = len(sketches[0])
        cdf = np.cumsum(n_pairs_arr)/sum(n_pairs_arr)
        ev_cdf = cdf**n_hash
        return ev_cdf
    
    def get_maximum_k_thresh(cdf, max_quantile=0.9):
        idx = np.flatnonzero(np.diff(cdf>max_quantile))[0]
        return k_vals[idx]

    def get_inflection_thresh(k_vals, cdf):
        log_cdf = np.log(np.gradient(cdf)+1)
        log_d2 = np.gradient(np.gradient(log_cdf))
        smooth_log_d2 = gaussian_filter1d(log_d2, 1)
        smooth_log_d2 = np.insert(smooth_log_d2, 0, 1000)
        infls = np.flatnonzero(np.diff(np.sign(smooth_log_d2)))
        k_thresh = k_vals[infls[1]]+1
        return k_thresh
    
    k_vals, n_pairs_arr = get_all_n_pairs_arr()
    cdf = get_extreme_value_cdf(n_pairs_arr)
    k_thresh = get_inflection_thresh(k_vals, cdf)
    maximum_k_thresh = get_maximum_k_thresh(cdf)
    k_thresh = min(k_thresh, maximum_k_thresh)
    return k_thresh

