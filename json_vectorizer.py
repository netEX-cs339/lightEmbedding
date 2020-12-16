import json
import pickle
import os.path as osp
import numpy as np
from jsonvectorizer import JsonVectorizer, vectorizers
from jsonvectorizer.utils import fopen

DATA_PATH = 'Data/json'
SAVE_PATH = 'Data/bin'
# Load data
docs = []
with fopen(osp.join(DATA_PATH, 'sample10000.json')) as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)


# Learn the schema of sample documents
vectorizer = JsonVectorizer()
vectorizer.extend(docs)

vectorizer.prune(patterns=['^_'], min_f=0.01)

# Report booleans as is
bool_vectorizer = {
    'type': 'boolean',
    'vectorizer': vectorizers.BoolVectorizer
}
'''
# For numbers, use one-hot encoding with 10 bins
number_vectorizer = {
    'type': 'number',
    'vectorizer': vectorizers.NumberVectorizer,
    'kwargs': {'n_bins': 10},
}
'''
# For numbers, use one-hot encoding with 10 bins
number_vectorizer = {
    'type': 'number',
    'vectorizer': vectorizers.NumberVectorizer,
    'kwargs': {'bins': 10},
}
# For strings use tokenization, ignoring sparse (<1%) tokens
string_vectorizer = {
    'type': 'string',
    'vectorizer': vectorizers.StringVectorizer,
    'kwargs': {'min_df': 0.01}
}

# Build JSON vectorizer
vectorizers = [
    bool_vectorizer,
    number_vectorizer,
    string_vectorizer
]

vectorizer.fit(vectorizers=vectorizers)

for i, feature_name in enumerate(vectorizer.feature_names_):
    print('{}: {}'.format(i, feature_name))

# Convert to CSR format for efficient row slicing
X = vectorizer.transform(docs).tocsr()

# Save
with open(osp.join(SAVE_PATH, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

np.save(osp.join(SAVE_PATH, "sample10000_bin.npy"), X.toarray())

