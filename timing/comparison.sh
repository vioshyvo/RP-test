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

if [ "$#" -eq 2 ]; then
  RESULT_FILE="results/$DATASET_NAME/mrpt_$2"
  RESULT_FILE_AUTO="results/$DATASET_NAME/mrpt_auto_$2"
else
  RESULT_FILE="results/$DATASET_NAME/mrpt"
  RESULT_FILE_AUTO="results/$DATASET_NAME/mrpt_auto"
fi

if [ $PARALLEL -eq 1 ]; then
  RESULT_FILE="${RESULT_FILE}_parallel"
  RESULT_FILE_AUTO="${RESULT_FILE_AUTO}_parallel"
  RESULT_FILE_OLD="results/$DATASET_NAME/mrpt_old_parallel"
else
  RESULT_FILE_OLD="results/$DATASET_NAME/mrpt_old"
fi

pushd mrpt_autotuning_tester
  make
popd

mrpt_autotuning_tester/tester $N $N_TEST $K $MRPT_AUTO_MAX_TREES $MRPT_AUTO_MIN_DEPTH $MRPT_AUTO_MAX_DEPTH $MRPT_AUTO_MAX_VOTES $DIM $MMAP "${RESULT_FILE_AUTO}_write" "data/$DATASET_NAME" "$MRPT_SPARSITY" "$PARALLEL" >> "$RESULT_FILE_AUTO"


# pushd mrpt_tester
#   make
# popd
#
# echo -n > "$RESULT_FILE"
# for n_trees in $MRPT_VOTING_N_TREES; do
#     for depth in $MRPT_DEPTH; do
#         mrpt_tester/tester $N $N_TEST $K $n_trees $depth $DIM $MMAP "results/$DATASET_NAME" "data/$DATASET_NAME" "$MRPT_SPARSITY" "$PARALLEL" $MRPT_VOTES  >> "$RESULT_FILE"
#     done
# done
#
# pushd mrpt_old_tester
#   make
# popd
#
# echo -n > "$RESULT_FILE_OLD"
# for n_trees in $MRPT_VOTING_N_TREES; do
#     for depth in $MRPT_DEPTH; do
#         mrpt_old_tester/tester $N $N_TEST $K $n_trees $depth $DIM $MMAP "results/$DATASET_NAME" "data/$DATASET_NAME" "$MRPT_SPARSITY" "$PARALLEL" $MRPT_VOTES  >> "$RESULT_FILE_OLD"
#     done
# done
