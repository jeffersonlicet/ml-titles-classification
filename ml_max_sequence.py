import numpy as np

titles = np.load('./output/titles_normal.npy', allow_pickle=True)
max_sequence = max([len(l) for l in titles])
print(max_sequence)
