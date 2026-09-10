#!/bin/bash
#SBATCH --job-name=sigml_16ag_NFF_loc0_loc9_ratios
#SBATCH --account=def-leil_cpu
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-49
#SBATCH --output=./ppo/%A_%a_16.log

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/main.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"
if [[ ! -f main.py ]]; then
    echo "[train_sigml_16ag_NFF] Could not locate main.py from $(pwd) -- submit from the project root." >&2
    exit 1
fi
if [[ ! -f venv/bin/activate ]]; then
    echo "[train_sigml_16ag_NFF] Could not find venv/bin/activate from $(pwd) -- venv/ may not exist here." >&2
    exit 1
fi

module load python/3.11
source ./venv/bin/activate

ALGO=ippo

# 2 locations x 5 ratios x 5 trials = 50 jobs
LOCS=(0 9)
RATIOS=(ratio_cap01_miss09 ratio_cap02_miss08 ratio_cap03_miss07 ratio_cap04_miss06 ratio_cap05_miss05)

IDX=${SLURM_ARRAY_TASK_ID}
TRIAL=$(( IDX % 5 ))
RATIO_IDX=$(( (IDX / 5) % 5 ))
LOC_IDX=$(( IDX / 25 ))

LOC=${LOCS[$LOC_IDX]}
RATIO=${RATIOS[$RATIO_IDX]}
SEED=${TRIAL}

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Algorithm: ${ALGO}"
echo "Location: ${LOC}"
echo "Ratio: ${RATIO}"
echo "Trial/Seed: ${SEED}"

python -u main.py \
  --env SIG \
  --loc $LOC \
  --algo $ALGO \
  --n_agent 16 \
  --seed $SEED \
  --trial_run $TRIAL \
  --env_config Configuration/env_ratios/${RATIO}.json \
  --train_data Environment/SUMOData/NFIG_k16.csv \
  --test_data Environment/SUMOData/NFIG_k16.csv
