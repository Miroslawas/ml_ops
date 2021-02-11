#!/usr/bin/env bash
out_path=$1
data_dir=$2
training_job_name=$(tail -n 1 $3)

python launch_batch_prediction_job.py --data-dir=${data_dir} --out-path=${out_path} --training-job-name=${training_job_name}
