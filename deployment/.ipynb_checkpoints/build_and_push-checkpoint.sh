#!/usr/bin/env bash

rm -r opt

mkdir -p opt
mkdir -p opt/ml/input/config
mkdir -p opt/ml/input/data
mkdir -p opt/ml/input/data/train
mkdir -p opt/ml/input/data/test
mkdir -p opt/ml/model
mkdir -p opt/ml/output/failure
mkdir -p opt/program

cp -r ../src/heart/src/models/* opt/program
mv opt/program/train_model.py opt/program/train
cp ../src/heart/src/models/params.json opt/ml/input/config
cp ../src/heart/data/processed/train.csv opt/ml/input/data/train
cp ../src/heart/data/processed/test.csv opt/ml/input/data/test

chmod -R 777 .

docker build  -t $1 .
