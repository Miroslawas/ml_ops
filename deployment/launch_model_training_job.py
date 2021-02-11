# S3 prefix
prefix = 'heart-sagemaker-demo'

# Define IAM role
import boto3
import re
import json

import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role

import argparse

import sagemaker as sage
from time import gmtime, strftime

role = get_execution_role()

sess = sage.Session()

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', dest='data_dir')
parser.add_argument('--prefix', dest='prefix')
parser.add_argument('--model', dest='model_name')
args = parser.parse_args()
    
WORK_DIRECTORY = args.data_dir
prefix = args.prefix

data_location = sess.upload_data(WORK_DIRECTORY, key_prefix=prefix)

metric_definitions = [
    {
        "Name": "roc_auc",
        "Regex": "roc_auc=(.*?)",
    },
    {
        "Name": "accuracy",
        "Regex": "accuracy=(.*?);",
    },
    {
        "Name": "lift",
        "Regex": "lift=(.*?);",
    },
    {
        "Name": "hosmer_lemeshow",
        "Regex": "hosmer_lemeshow=(.*?);",
    },
]


account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, args.model_name)

tree = sage.estimator.Estimator(image,
                                role, 1, 'ml.c4.2xlarge',
                                output_path="s3://{}/output".format(sess.default_bucket()),
                                sagemaker_session=sess,
                                tags=[{'Key': 'CostGroup', 'Value': 'DEV-TA-ANALYTICS'}],
                                )

print('training model')
tree.fit(data_location, {'training': data_location})

print(tree.latest_training_job.name)
