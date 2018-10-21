#!/usr/bin/env bash

if [ ! -f "parameters/$1.sh" ]; then
    echo Invalid data set 1>&2
    exit
fi

DATASET_NAME="$1"
. "parameters/$DATASET_NAME.sh"
. "data/$DATASET_NAME/dimensions.sh"


# N=60000
# N_TEST=100
K=10
# DIM=784
# MMAP=0
# DATASET_NAME=mnist

pushd exact
  make
popd

mkdir -p "results/$DATASET_NAME"
exact/tester $N $N_TEST $K $DIM $MMAP "data/$DATASET_NAME" > "results/$DATASET_NAME/truth_$K"
