#!/bin/bash
#SBATCH --workdir=/wrk/hyvi/RP-test/timing
#SBATCH --job-name=timing-mnist-mrpt-full
#SBATCH -o script-output/timing-mnist-mrpt-full.txt
#SBATCH -c 1
#SBATCH -p test
#SBATCH -t 00:10:00
#SBATCH --mem=2G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if [ $# -ne 1 ]; then
  echo "Usage ./timing-mnist-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

srun ./timing_comparison.sh mnist1000 "$1"
