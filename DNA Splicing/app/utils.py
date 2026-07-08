import numpy as np

def encode_dna(seq):
    mapping = {
        "A": [1,0,0],
        "C": [0,1,0],
        "G": [0,0,1],
        "T": [0,0,0]
    }

    max_len = 60

    seq = seq[:max_len]

    if len(seq) < max_len:
        seq += "A" * (max_len - len(seq))

    encoded = []

    for base in seq:
        encoded.extend(mapping.get(base, [0,0,0]))

    return np.array(encoded)