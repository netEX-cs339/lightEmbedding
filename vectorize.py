import json
import os
import pickle
import os.path as osp
import numpy as np
from jsonvectorizer import JsonVectorizer, vectorizers
from jsonvectorizer.utils import fopen

DATA_PATH = 'Data/json'
SAVE_PATH = 'Data/bin'

# Load data
docs = []
with fopen(osp.join(DATA_PATH, 'sample2000.json')) as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)

with open('Data/bin/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Convert to CSR format for efficient row slicing
X = vectorizer.transform(docs).tocsr()
X_arr = X.toarray()
print(np.shape(X_arr))

# Save
np.save(osp.join(SAVE_PATH, "sample2000_bin.npy"), X_arr)
np.save(osp.join(SAVE_PATH, "dim.npy"), X_arr.shape[1])
