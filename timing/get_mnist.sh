#!/usr/bin/env bash

REMOVE_DOWNLOADED=false # remove downloaded datasets after they've been converted
N=60000
TEST_N=100 # number of test queries
DIM=784


DATA_DIR="data"
MNIST_DIR="$DATA_DIR/mnist"
if [ ! -f "$MNIST_DIR/data.bin" ]; then
    mkdir -p data/mnist
    echo "Downloading MNIST..."
    curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    echo "Extracting MNIST..."
    gunzip train-images-idx3-ubyte.gz
    echo "Converting MNIST..."
    python2 tools/binary_converter.py train-images-idx3-ubyte "$MNIST_DIR/data.bin"
    python2 tools/binary_converter.py --sample "$MNIST_DIR/data.bin" "$MNIST_DIR/train.bin" "$MNIST_DIR/test.bin" "$TEST_N" "$DIM"
    if [ "$REMOVE_DOWNLOADED" = true ]; then
        rm train-images-idx3-ubyte
    fi

    DIM_FILE="$MNIST_DIR/dimensions.sh"
    echo '#!/usr/bin/env bash' > "$DIM_FILE"
    echo >> "$DIM_FILE"
    echo N="$N" >> "$DIM_FILE"
    echo N_TEST="$TEST_N" >> "$DIM_FILE"
    echo DIM="$DIM" >> "$DIM_FILE"
fi
