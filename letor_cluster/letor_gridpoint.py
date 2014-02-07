#!/usr/bin/env python
import sys
import os
import json
from time import time
import numpy as np

from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor


def dcg(relevances, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)
 
 
def ndcg(relevances, rank=10):
    """Normalized discounted cumulative gain (NDGC)"""
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(relevances, rank) / best_dcg


def mean_ndcg(y_true, y_pred, query_ids, rank=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    query_ids = np.asarray(query_ids)
    # assume query_ids are sorted
    ndcg_scores = []
    previous_qid = query_ids[0]
    previous_loc = 0
    for loc, qid in enumerate(query_ids):
        if previous_qid != qid:
            chunk = slice(previous_loc, loc)
            ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
            ndcg_scores.append(ndcg(ranked_relevances, rank=rank))
            previous_loc = loc
        previous_qid = qid

    chunk = slice(previous_loc, loc + 1)
    ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
    ndcg_scores.append(ndcg(ranked_relevances, rank=rank))
    return np.mean(ndcg_scores)


job_folder = sys.argv[1]
with open(job_folder + '/parameters.json', 'r') as f:
    parameters = json.load(f)

with open(job_folder + '/data.json', 'r') as f:
    data_filenames = json.load(f)


print("Loading the data...")
tic = time()
X_train, y_train, qid_train = joblib.load(data_filenames['train'],
                                          mmap_mode='r')
X_vali, y_vali, qid_vali = joblib.load(data_filenames['validation'],
                                       mmap_mode='r')
# warm up (load the data from the drive)
X_train.max(), X_vali.max()
data_load_time = time() - tic
print("done in{:.3f}s".format(data_load_time))

print("Training the model with parameters:")
print(parameters)
tic = time()
model = GradientBoostingRegressor(random_state=0)
model.set_params(**parameters)
model.fit(X_train, y_train)
training_time = time() - tic
print("done in{:.3f}s".format(training_time))

print("Computing training NDGC@10...")
tic = time()
y_pred = model.predict(X_train)
prediction_time = time() - tic
train_score = mean_ndcg(y_train, y_pred, qid_train)
print("{:.3f}".format(train_score))
print("done in{:.3f}s".format(prediction_time))

print("Computing validation NDGC@10...")
y_pred = model.predict(X_vali)
validation_score = mean_ndcg(y_vali, y_pred, qid_vali)
print("{:.3f}".format(validation_score))

model_filename = job_folder + '/model.pkl'
print("Saving model to {}".format(model_filename))
tic = time()
model_filenames = joblib.dump(model, model_filename)
model_save_time = time() - tic
print("done in{:.3f}s".format(model_save_time))
model_size_bytes = 0
for filename in model_filenames:
    model_size_bytes += os.stat(filename).st_size

results = {
    'data_load_time': data_load_time,
    'training_time': training_time,
    'prediction_time': prediction_time,
    'model_save_time': model_save_time,
    'model_size_bytes': model_size_bytes,
    'train_score': train_score,
    'validation_score': validation_score,
    'model_filename': model_filename,
}

with open(job_folder + '/results.json', 'wb') as f:
    f.write(json.dumps(results).encode('utf-8'))
