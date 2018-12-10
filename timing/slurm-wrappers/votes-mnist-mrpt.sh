#!/bin/bash
#SBATCH --workdir=/wrk/hyvi/RP-test/timing
#SBATCH --job-name=votes-mnist-mrpt-full
#SBATCH -o script-output/votes-mnist-mrpt-full.txt
#SBATCH -c 1
#SBATCH -t 02:00:00
#SBATCH --mem=5G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if [ $# -ne 1 ]; then
  echo "Usage ./votes-mnist-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

srun ./votes_comparison.sh mnist "$1"
