#!/bin/bash
#SBATCH --workdir=/wrk/hyvi/RP-test/timing
#SBATCH --job-name=timing-sift-mrpt-default
#SBATCH -o script-output/timing-sift-mrpt-default.txt
#SBATCH -c 1
#SBATCH -t 20:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if [ $# -ne 1 ]; then
  echo "Usage ./timing-sift-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

srun ./timing_comparison.sh sift1000 "$1"
