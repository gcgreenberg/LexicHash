import os
import pyfastx

BASES = {'A','T','C','G'}
RC = {'A':'T','T':'A','C':'G','G':'C','a':'t','t':'a','c':'g','g':'c'}
revcomp = lambda seq: ''.join([RC[b] for b in seq[::-1]])
revcomp_seqs = lambda seqs: [revcomp(seq) for seq in seqs]

def clean_seqs(seqs, replace_nonbase=''):
    for i in range(len(seqs)):
        seq = seqs[i]
        for b in BASES: seq = seq.replace(b.lower(), b)
        for char in set(seq).difference({'A','T','C','G'}):
            seq = seq.replace(char, replace_nonbase)
        seqs[i] = seq
    return seqs
    
def get_fasta(file):
    return pyfastx.Fasta(file)

def get_fastq(file):
    return pyfastx.Fastq(file)

def get_seqs(file, return_ids=False):
    seqs = pyfastx.Fastx(file)
    seqs = list(seqs)
    seqs,ids = [s[1] for s in seqs], [s[0] for s in seqs]
#    seqs = []
#    ids = []
#    with open(file, 'r') as fh:
#        for line in fh.readlines():
#            if line[0] == ">" or line[0] == '@': 
#                seqs.append('')
#                ids.append(line[1:].split()[0])
#            else: seqs[-1] += line
    seqs = clean_seqs(seqs)
    if return_ids:
        return seqs,ids
    else:
        return seqs

def get_seq(file, seq_id=None, longest=False): # default is first seq
    f = get_fasta(file)
    return f.longest.seq if longest else f[seq_id].seq if seq_id else f[0].seq
    
def get_longest_seq_id(file): # seq id of first seq
    f = get_fasta(file)
    return f.longest.name

########### WRITING FILES

def write_fasta(file, seqs): 
    '''seqs should be list of tuples (id, seq)'''
    with open(file, 'w') as fh:
        for seq in seqs:
            fh.write('>' + seq[0] + "\n")
            fh.write(seq[1] + "\n")
            
            
#########################################
