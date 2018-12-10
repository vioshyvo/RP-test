#!/bin/bash
#SBATCH --workdir=/wrk/hyvi/RP-test/timing
#SBATCH --job-name=grid-gist-mrpt-full
#SBATCH -o script-output/grid-gist-mrpt-full.txt
#SBATCH -c 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=30G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if [ $# -ne 1 ]; then
  echo "Usage ./grid-gist-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

srun ./normal_comparison.sh gist "$1"
