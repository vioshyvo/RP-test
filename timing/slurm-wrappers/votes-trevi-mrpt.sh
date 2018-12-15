#!/bin/bash
#SBATCH --workdir=/wrk/hyvi/RP-test/timing
#SBATCH --job-name=votes-trevi-mrpt-full
#SBATCH -o script-output/votes-trevi-mrpt-full.txt
#SBATCH -c 1
#SBATCH -t 04-00:00:00
#SBATCH --mem=30G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ville.o.hyvonen@helsinki.fi

if [ $# -ne 1 ]; then
  echo "Usage ./votes-trevi-mrpt.sh <post-fix>"
  exit
fi

module load GCCcore/7.3.0

srun ./votes_comparison.sh trevi "$1"
