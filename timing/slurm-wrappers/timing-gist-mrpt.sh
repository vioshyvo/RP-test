#!/bin/bash
#SBATCH --workdir=/wrk/hyvi/RP-test/timing
#SBATCH --job-name=timing-gist-mrpt
#SBATCH -o script-output/timing-gist-mrpt.txt
#SBATCH -c 1
#SBATCH -t 10:00:00
#SBATCH --mem=30G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if [ $# -ne 1 ]; then
  echo "Usage ./timing-gist-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

srun ./timing_comparison.sh gist "$1"
