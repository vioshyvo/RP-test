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

RESULT_DIR="results/times_${DATASET_NAME}"
mkdir -p "$RESULT_DIR"

pushd exact
  make
popd

for K in 1 10 100; do
  if [ ! -f "$RESULT_DIR/truth_$K" ]; then
    ./exact/tester $N $N_TEST $K $DIM $MMAP "data/$DATASET_NAME">"$RESULT_DIR/truth_$K"
  fi
done


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

echo -n > "$RESULT_FILE_AUTO"
timing_tester/tester $N $N_TEST $K $MRPT_AUTO_MAX_TREES $MRPT_AUTO_MIN_DEPTH $MRPT_AUTO_MAX_DEPTH $MRPT_AUTO_MAX_VOTES $DIM $MMAP "$RESULT_DIR" "data/$DATASET_NAME" "$MRPT_SPARSITY" "$PARALLEL" "$N_AUTO" "$RESULT_FILE_AUTO"
