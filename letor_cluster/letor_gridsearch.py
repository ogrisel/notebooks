import numpy as np
import os
from random import Random
import json

from sklearn.externals import joblib
from sklearn.grid_search import ParameterGrid


MSLR_DATA = '/scratch/ogrisel/mslr-web10k_fold1.npz'
DATA_FOLDER = '/home/parietal/ogrisel/data'
TRAIN_SAMPLE_DATA = DATA_FOLDER + '/mslr-web10k_fold1_train_500.pkl'
VALI_DATA = DATA_FOLDER + '/mslr-web10k_fold1_vali.pkl'
GRID_JOBS_FOLDER = '/scratch/ogrisel/grid_jobs'

rng = Random(42)


def subsample(X, y, qid, size, seed=None):
    rng = np.random.RandomState(seed)
    unique_qid = np.unique(qid)
    qid_mask = rng.permutation(len(unique_qid))[:size]
    subset_mask = np.in1d(qid_train, unique_qid[qid_mask])
    return X[subset_mask], y[subset_mask], qid[subset_mask]


if not os.path.exists(TRAIN_SAMPLE_DATA) or not os.path.exists(VALI_DATA):
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    data = np.load(os.path.expanduser(MSLR_DATA))
    X_train, y_train, qid_train = data['X_train'], data['y_train'], data['qid_train']
    X_vali, y_vali, qid_vali = data['X_vali'], data['y_vali'], data['qid_vali']

    X_train_small, y_train_small, qid_train_small = subsample(
        X_train, y_train, qid_train, 500, seed=0)

    joblib.dump((X_train_small, y_train_small, qid_train_small),
                TRAIN_SAMPLE_DATA)
    joblib.dump((X_vali, y_vali, qid_vali), VALI_DATA)


if not os.path.exists(GRID_JOBS_FOLDER):
    os.makedirs(GRID_JOBS_FOLDER)


params = {
    'max_features': [10, 20, 50, 100],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.5, 0.8, 1.0],
    'loss': ['ls', 'huber', 'quantile'],
    'learning_rate': [0.05, 0.1, 0.5],
}

for i, param in enumerate(ParameterGrid(params)):
    params_description = json.dumps(param)
    job_id = joblib.hash(params_description)
    job_folder = GRID_JOBS_FOLDER + '/' + job_id
    if not os.path.exists(job_folder):
        os.makedirs(job_folder)
    with open(job_folder + '/parameters.json', 'wb') as f:
        f.write(params_description.encode('utf-8'))

    data_filenames = {'train': TRAIN_SAMPLE_DATA, 'validation': VALI_DATA}
    with open(job_folder + '/data.json', 'wb') as f:
        f.write(json.dumps(data_filenames).encode('utf-8'))

    cmd = 'qsub -V -cwd letor_gridpoint.py {}'.format(job_folder)
    os.system(cmd)

    # if i > 100:
    #     break