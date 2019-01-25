#!/bin/bash
#SBATCH --workdir=/wrk/hyvi/RP-test/timing
#SBATCH --job-name=adaptive-mnist-mrpt-full
#SBATCH -o script-output/grid-mnist-mrpt-full.txt
#SBATCH -c 1
#SBATCH -t 20:00:00
#SBATCH --mem=5G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if [ $# -ne 1 ]; then
  echo "Usage ./adaptive-mnist-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

srun ./grid_comparison.sh mnist "$1"
