#!/usr/bin/env bash

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
   echo "error: Expecting parameters: <data-set-name> or <data-set-name> <postfix>" 1>&2
   exit
fi


if [ ! -f "parameters/$1.sh" ]; then
    echo Invalid data set 1>&2
    exit
fi

DATASET_NAME="$1"

. "parameters/$DATASET_NAME.sh"
. "data/$DATASET_NAME/dimensions.sh"

pushd exact
  make
popd

mkdir -p "results/$DATASET_NAME"
for K in 1 10 100; do
  if [ ! -f "results/$DATASET_NAME/truth_$K" ]; then
    ./exact/tester $N $N_TEST $K $DIM $MMAP "data/$DATASET_NAME" > "results/$DATASET_NAME/truth_$K"
  fi
done

pushd mrpt_tester
  make
popd

if [ "$#" -eq 2 ]; then
  RESULT_FILE="results/$DATASET_NAME/mrpt_$2"
else
  RESULT_FILE="results/$DATASET_NAME/mrpt"
fi

echo -n > "$RESULT_FILE"
for n_trees in $MRPT_VOTING_N_TREES; do
    for depth in $MRPT_DEPTH; do
        mrpt_tester/tester $N $N_TEST $K $n_trees $depth $DIM $MMAP "results/$DATASET_NAME" "data/$DATASET_NAME" "$MRPT_SPARSITY" $MRPT_VOTES  >> "$RESULT_FILE"
    done
done

pushd mrpt_old_tester
  make
popd

echo -n > "results/$DATASET_NAME/mrpt_old"
for n_trees in $MRPT_VOTING_N_TREES; do
    for depth in $MRPT_DEPTH; do
        mrpt_old_tester/tester $N $N_TEST $K $n_trees $depth $DIM $MMAP "results/$DATASET_NAME" "data/$DATASET_NAME" "$MRPT_SPARSITY" $MRPT_VOTES  >> "results/$DATASET_NAME/mrpt_old"
    done
done
