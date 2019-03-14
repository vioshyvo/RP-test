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

if [ ! -f "data/$DATASET_NAME/dimensions.sh" ]; then
  echo "Data set $DATASET_NAME not yet downloaded or converted to binary."
  exit
fi

. "parameters/$DATASET_NAME.sh"
. "data/$DATASET_NAME/dimensions.sh"

pushd exact
  make
popd

mkdir -p "results/$DATASET_NAME"
for K in 1 10 100; do
  if [ ! -f "results/$DATASET_NAME/truth_$K" ]; then
    ./exact/tester $N $N_TEST $K $DIM $MMAP "data/$DATASET_NAME">"results/$DATASET_NAME/truth_$K"
  fi
done

if [ "$#" -eq 2 ]; then
  RESULT_FILE="results/$DATASET_NAME/mrpt_$2"
else
  RESULT_FILE="results/$DATASET_NAME/mrpt"
fi

if [ $PARALLEL -eq 1 ]; then
  RESULT_FILE="${RESULT_FILE}_parallel"
fi

pushd mrpt2_tester
  make
popd

echo -n > "$RESULT_FILE"
for n_trees in $MRPT_VOTING_N_TREES; do
    for depth in $MRPT_DEPTH; do
        mrpt2_tester/tester $N $N_TEST $K $n_trees $depth $DIM $MMAP "results/$DATASET_NAME" "data/$DATASET_NAME" "$MRPT_SPARSITY" "$PARALLEL" $MRPT_VOTES >> "$RESULT_FILE"
    done
done
