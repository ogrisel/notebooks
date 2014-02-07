import pandas as pd
import os
import os.path as op
import json


def collect_results(jobs_folder):
    entries = []

    for job_folder in os.listdir(jobs_folder):
        results_filename = op.join(
            jobs_folder, job_folder, 'results.json')
        parameters_filename = op.join(
            jobs_folder, job_folder, 'parameters.json')

        if (not op.exists(parameters_filename)
                or not op.exists(results_filename)):
            continue

        new_entry = dict()

        with open(parameters_filename, 'r') as f:
            new_entry.update(json.load(f))

        with open(results_filename, 'r') as f:
            new_entry.update(json.load(f))

        entries.append(new_entry)

    return pd.DataFrame(entries)


if __name__ == '__main__':
    results = collect_results('/scratch/ogrisel/grid_jobs')
    results.to_json('letor_gridresults.json')