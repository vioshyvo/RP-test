#!/usr/bin/env bash

REMOVE_DOWNLOADED=false # remove downloaded datasets after they've been converted
TEST_N=1000 # number of test queries
DATA_DIR="data"

MNIST_N=60000
MNIST_DIM=784
MNIST_DIR="$DATA_DIR/mnist1000"

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
    python2 tools/binary_converter.py train-images-idx3-ubyte "$MNIST_DIR/data.bin"
    python2 tools/binary_converter.py --sample "$MNIST_DIR/data.bin" "$MNIST_DIR/train.bin" "$MNIST_DIR/test.bin" "$TEST_N" "$MNIST_DIM"
    if [ "$REMOVE_DOWNLOADED" = true ]; then
        rm train-images-idx3-ubyte
    fi

    MNIST_DIM_FILE="$MNIST_DIR/dimensions.sh"
    echo '#!/usr/bin/env bash' > "$MNIST_DIM_FILE"
    echo >> "$MNIST_DIM_FILE"
    echo N="$MNIST_N" >> "$MNIST_DIM_FILE"
    echo N_TEST="$TEST_N" >> "$MNIST_DIM_FILE"
    echo DIM="$MNIST_DIM" >> "$MNIST_DIM_FILE"
fi

SIFT_N=1000000
SIFT_DIM=128
SIFT_DIR="$DATA_DIR/sift1000"

if [ ! -f "$SIFT_DIR/data.bin" ]; then
    mkdir -p "$SIFT_DIR"
    if [ ! -f  sift/sift_base.fvecs ]; then
      echo "Downloading SIFT..."
      wget "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" -O sift.tar.gz
      echo "Extracting SIFT..."
    else
      echo "SIFT already downloaded, using cached version..."
    fi
    tar -zxf sift.tar.gz
    echo "Converting SIFT..."
    python2 tools/binary_converter.py sift/sift_base.fvecs "$SIFT_DIR/data.bin" "$SIFT_N"
    python2 tools/binary_converter.py --sample "$SIFT_DIR/data.bin" "$SIFT_DIR/train.bin" "$SIFT_DIR/test.bin" "$TEST_N" "$SIFT_DIM"
    if [ "$REMOVE_DOWNLOADED" = true ]; then
        rm sift/*
        rm sift.tar.gz
    fi

    SIFT_DIM_FILE="$SIFT_DIR/dimensions.sh"
    echo '#!/usr/bin/env bash' > "$SIFT_DIM_FILE"
    echo >> "$SIFT_DIM_FILE"
    echo N="$SIFT_N" >> "$SIFT_DIM_FILE"
    echo N_TEST="$TEST_N" >> "$SIFT_DIM_FILE"
    echo DIM="$SIFT_DIM" >> "$SIFT_DIM_FILE"
fi

GIST_N=1000000
GIST_DIM=960
GIST_DIR="$DATA_DIR/gist1000"


if [ ! -f "$GIST_DIR/data.bin" ]; then
    mkdir -p "$GIST_DIR"
    if [ ! -f  gist/gist_base.fvecs ]; then
      echo "Downloading GIST..."
      wget "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" -O gist.tar.gz
      echo "Extracting GIST..."
      tar xzf gist.tar.gz
    else
      echo "GIST already downloaded, using cached version..."
    fi
    echo "Converting GIST..."
    python2 tools/binary_converter.py gist/gist_base.fvecs "$GIST_DIR/data.bin"
    python2 tools/binary_converter.py --sample "$GIST_DIR/data.bin" "$GIST_DIR/train.bin" "$GIST_DIR/test.bin" $TEST_N "$GIST_DIM"
    if [ "$REMOVE_DOWNLOADED" = true ]; then
        rm -r gist
        rm gist.tar.gz
    fi

    GIST_DIM_FILE="$GIST_DIR/dimensions.sh"
    echo '#!/usr/bin/env bash' > "$GIST_DIM_FILE"
    echo >> "$GIST_DIM_FILE"
    echo N="$GIST_N" >> "$GIST_DIM_FILE"
    echo N_TEST="$TEST_N" >> "$GIST_DIM_FILE"
    echo DIM="$GIST_DIM" >> "$GIST_DIM_FILE"
fi

STL_N=100000
STL_DIM=9216
STL_DIR="$DATA_DIR/stl10"


if [ ! -f "$STL_DIR/data.bin" ]; then
    mkdir -p "$STL_DIR"
    if [ ! -f stl10_binary.tar.gz ]; then
      echo "Downloading STL-10..."
      wget "http://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz" -O stl10_binary.tar.gz
    else
      echo "STL-10 already downloaded, using cached version..."
    fi

    echo "Extracting STL-10..."
    tar xzf stl10_binary.tar.gz

    echo "Converting STL-10..."
    python2 tools/binary_converter.py stl10_binary/unlabeled_X.bin "$STL_DIR/data.bin"
    python2 tools/binary_converter.py --sample "$STL_DIR/data.bin" "$STL_DIR/train.bin" "$STL_DIR/test.bin" "$TEST_N" "$STL_DIM"
    rm -r stl10_binary
    if [ "$REMOVE_DOWNLOADED" = true ]; then
      rm stl10_binary.tar.gz
    fi

    STL_DIM_FILE="$STL_DIR/dimensions.sh"
    echo '#!/usr/bin/env bash' > "$STL_DIM_FILE"
    echo >> "$STL_DIM_FILE"
    echo N="$STL_N" >> "$STL_DIM_FILE"
    echo N_TEST="$TEST_N" >> "$STL_DIM_FILE"
    echo DIM="$STL_DIM" >> "$STL_DIM_FILE"
fi


TREVI_N=101120
TREVI_DIM=4096
TREVI_DIR="$DATA_DIR/trevi"

if [ ! -f "$TREVI_DIR/data.bin" ]; then
    mkdir -p "$TREVI_DIR"
    if [ ! -f trevi.zip ]; then
      echo "Downloading Trevi..."
      wget "http://phototour.cs.washington.edu/patches/trevi.zip" -O trevi.zip
    else
      echo "Trevi already downloaded, using cached version..."
    fi

    echo "Extracting Trevi..."
    mkdir patches
    unzip -q trevi.zip -d patches

    echo "Converting Trevi..."
    python2 tools/binary_converter.py patches/ "$TREVI_DIR/data.bin"
    python2 tools/binary_converter.py --sample "$TREVI_DIR/data.bin" "$TREVI_DIR/train.bin" "$TREVI_DIR/test.bin" "$TEST_N" "$TREVI_DIM"
    rm -r patches
    if [ "$REMOVE_DOWNLOADED" = true ]; then
        rm trevi.zip
    fi

    TREVI_DIM_FILE="$TREVI_DIR/dimensions.sh"
    echo '#!/usr/bin/env bash' > "$TREVI_DIM_FILE"
    echo >> "$TREVI_DIM_FILE"
    echo N="$TREVI_N" >> "$TREVI_DIM_FILE"
    echo N_TEST="$TEST_N" >> "$TREVI_DIM_FILE"
    echo DIM="$TREVI_DIM" >> "$TREVI_DIM_FILE"
fi
