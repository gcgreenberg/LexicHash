
# LexicHash

### Sequence Similarity Estimation via Lexicographic Comparison of Hashes

For help running LexicHash/MinHash, run the following in LexicHash directory:

`python3 run_module.py -h`

-----

Example on NCTC1080 dataset:

`python3 run_module.py --out ../LH_out --fasta data/NCTC1080/NCTC1080_reads.fasta.gz --n_hash 100 --max_k 32`

-----

To install requirements, run

`pip install -r requirements.txt`

-----

For now, LexicHash cannot deal with sequences with N's.

-----

The groundtruth alignmnent file has the following column format:

1. First read index (zero-based)
2. Second read index (zero-based)
3. Alignment size in base-pairs
4. Second read alignment orientation (+/-)

The output alignmnent file has the following column format:

1. First read index (zero-based)
2. Second read index (zero-based)
3. Alignment score
4. Second read alignment orientation (+/-)

