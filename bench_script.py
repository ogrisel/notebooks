import os
import json
import sys
import joblib
import threadpoolctl
import joblib
from time import perf_counter
from sklearn.metrics import log_loss


MIN_DURATION_SECONDS = 3.0

params = json.loads(sys.argv[1])
module = __import__(params["module"], fromlist=[params["estimator"]])
estimator_class = getattr(module, params["estimator"])
X, y = joblib.load(params["data_filepath"])

if "threadpool_limits" in params:
    threadpoolctl.threadpool_limits(limits=params["threadpool_limits"])

est = estimator_class()

tic = perf_counter()
count = 0
while perf_counter() - tic < MIN_DURATION_SECONDS:
    count += 1
    est.fit(X, y)
toc = perf_counter()

duration = (toc - tic) / count

log_loss_value = log_loss(y, est.predict_proba(X))


results = {
    "cpu_count": joblib.cpu_count(),
    "cpu_count_physical": joblib.cpu_count(only_physical_cores=True),
    "threadpool_info": threadpoolctl.threadpool_info(),
    "fit_time_seconds": duration,
    "n_iterations": count,
    "n_samples": X.shape[0],
    "n_features": X.shape[1],
    "n_classes": est.classes_.shape[0],
    "log_loss": log_loss_value,
}
with open(f"results_{params.get('run_id', 'default')}.json", "w") as f:
    json.dump(results, f)
