#!/bin/bash
#SBATCH --job-name=mnist-mrpt
#SBATCH -o ../script-output/mnist-mrpt.txt
#SBATCH -c 1
#SBATCH -p test
#SBATCH -t 00:30:00
#SBATCH --mem=2G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if[ $# -ne 1 ]; then
  echo "Usage ./mnist-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

cd ..
cd ann-benchmarks
srun ./comparison.sh mnist "$1"
