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

RESULT_DIR="results/times_${DATASET_NAME}"
mkdir -p "$RESULT_DIR"


if [ "$#" -eq 2 ]; then
  RESULT_FILE_AUTO="$RESULT_DIR/mrpt_times_$2"
else
  RESULT_FILE_AUTO="$RESULT_DIR/mrpt_times"
fi

if [ $PARALLEL -eq 1 ]; then
  RESULT_FILE_AUTO="${RESULT_FILE_AUTO}_parallel"
fi

pushd timing_tester
  make
popd

K=10
echo -n > "$RESULT_FILE_AUTO"
timing_tester/tester $N $N_TEST $K $MRPT_AUTO_MAX_TREES $MRPT_AUTO_MIN_DEPTH $MRPT_AUTO_MAX_DEPTH $MRPT_AUTO_MAX_VOTES $DIM $MMAP "$RESULT_DIR" "data/$DATASET_NAME" "$MRPT_SPARSITY" "$PARALLEL" >> "$RESULT_FILE_AUTO"
