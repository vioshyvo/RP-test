#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
   echo "error: Expecting parameters: <n_train> <n_test>" 1>&2
   exit
fi

N_TRAIN=$1
N_TEST=$2 # number of test queries
N=$(( $N_TRAIN + $N_TEST ))
DATA_DIR="data"

MNIST_DIM=784
MNIST_DIR="$DATA_DIR/mnist$N_TRAIN"

if [ ! -f "$MNIST_DIR/data.bin" ]; then
    mkdir -p "$MNIST_DIR"
    if [ ! -f train-images-idx3-ubyte ]; then
      echo "Downloading MNIST..."
      curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
      echo "Extracting MNIST..."
      gunzip train-images-idx3-ubyte.gz
    else
      echo "MNIST already downloaded, using cached version..."
    fi
    echo "Converting MNIST..."
    python tools/binary_converter.py train-images-idx3-ubyte "$MNIST_DIR/data.bin"
    python tools/binary_converter.py --sample "$MNIST_DIR/data.bin" "$MNIST_DIR/train.bin" "$MNIST_DIR/test.bin" "$N_TEST" "$MNIST_DIM" "$N_TRAIN"

    MNIST_DIM_FILE="$MNIST_DIR/dimensions.sh"
    echo '#!/usr/bin/env bash' > "$MNIST_DIM_FILE"
    echo >> "$MNIST_DIM_FILE"
    echo N="$N" >> "$MNIST_DIM_FILE"
    echo N_TEST="$N_TEST" >> "$MNIST_DIM_FILE"
    echo DIM="$MNIST_DIM" >> "$MNIST_DIM_FILE"
fi
