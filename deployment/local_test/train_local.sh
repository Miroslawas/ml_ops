#!/bin/sh

image=$1

rm -r test_dir/model
rm -r test_dir/output

mkdir -p test_dir/model
mkdir -p test_dir/output

docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
