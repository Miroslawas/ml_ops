# Define IAM role
import re
import json

import os
from sagemaker import get_execution_role

import argparse

import sagemaker as sage
from time import gmtime, strftime

role = get_execution_role()

sess = sage.Session()

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', dest='data_dir')
parser.add_argument('--out-path', dest='output_path')
parser.add_argument('--training-job-name', dest='training_job_name')
args = parser.parse_args()
    

estimator = sage.estimator.Estimator.attach(args.training_job_name)
output_path="s3://{}/{}".format(sess.default_bucket(), args.output_path + '-out')

transformer = estimator.transformer(instance_count=1,
                               instance_type='ml.m4.xlarge',
                               output_path=output_path,
                               assemble_with='Line',
                               accept='text/csv')

data_location = sess.upload_data(args.data_dir, key_prefix=args.output_path)

transformer.transform(data_location, content_type='text/csv', split_type='Line', input_filter='$[:-1]', logs=True)
transformer.wait()
