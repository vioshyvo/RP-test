#!/bin/bash
#SBATCH --workdir=/wrk/hyvi/RP-test/timing
#SBATCH --job-name=votes-sift-mrpt-full
#SBATCH -o script-output/votes-sift-mrpt-full.txt
#SBATCH -c 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if [ $# -ne 1 ]; then
  echo "Usage ./votes-sift-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

srun ./votes_comparison.sh sift "$1"
