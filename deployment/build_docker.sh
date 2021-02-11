#!/usr/bin/env bash

mkdir -p opt
mkdir -p opt/ml/input/config
mkdir -p opt/ml/input/data
mkdir -p opt/ml/input/data/train
mkdir -p opt/ml/input/data/test
mkdir -p opt/ml/model
mkdir -p opt/ml/output/failure
mkdir -p opt/program

cp -r ../src/* opt/program
mv opt/program/train_model.py opt/program/train

chmod -R 777 .

docker build  -t $1 .

rm -r opt