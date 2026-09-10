#!/bin/bash
#SBATCH --job-name=sigml_16ag_NFF_loc0_loc9_ratios_resume
#SBATCH --account=def-leil_cpu
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-49
#SBATCH --output=./ppo/%A_%a_16_resume.log

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/main.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"
if [[ ! -f main.py ]]; then
    echo "[train_sigml_16ag_NFF_resume] Could not locate main.py from $(pwd) -- submit from the project root." >&2
    exit 1
fi
if [[ ! -f venv/bin/activate ]]; then
    echo "[train_sigml_16ag_NFF_resume] Could not find venv/bin/activate from $(pwd) -- venv/ may not exist here." >&2
    exit 1
fi

module load python/3.11
source ./venv/bin/activate

ALGO=ippo

# Same 2 locations x 5 ratios x 5 trials = 50 jobs, same task-id -> (loc, ratio,
# trial) mapping as train_sigml_16ag_NFF.sh, so each array task resumes the
# checkpoint its non-resume counterpart produced.
LOCS=(0 9)
RATIOS=(ratio_cap01_miss09 ratio_cap02_miss08 ratio_cap03_miss07 ratio_cap04_miss06 ratio_cap05_miss05)
CAP_MISS=(cap0.1_miss0.9 cap0.2_miss0.8 cap0.3_miss0.7 cap0.4_miss0.6 cap0.5_miss0.5)

IDX=${SLURM_ARRAY_TASK_ID}
TRIAL=$(( IDX % 5 ))
RATIO_IDX=$(( (IDX / 5) % 5 ))
LOC_IDX=$(( IDX / 25 ))

LOC=${LOCS[$LOC_IDX]}
RATIO=${RATIOS[$RATIO_IDX]}
TAG=${CAP_MISS[$RATIO_IDX]}
SEED=${TRIAL}

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Algorithm: ${ALGO}"
echo "Location: ${LOC}"
echo "Ratio: ${RATIO}"
echo "Trial/Seed: ${SEED}"

# Find this task's checkpoint -- glob because the run timestamp embedded in
# the filename isn't known ahead of time. Picks the highest-episode
# checkpoint on disk (the run's most recent save, not necessarily ep20000 if
# training had already gone further than that for some task).
CKPT_DIR="$SCRIPT_DIR/Results/IPPO/checkpoints"
CKPT_PATTERN="${CKPT_DIR}/IPPO_SIG-SL_loc${LOC}_16ag_4sc_NFF_${TAG}_trial${TRIAL}_*_ep*.pt"
CKPT=$(ls -1 ${CKPT_PATTERN} 2>/dev/null | sed -E 's/.*_ep([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)

if [[ -z "$CKPT" ]]; then
    echo "[train_sigml_16ag_NFF_resume] No checkpoint found matching ${CKPT_PATTERN} -- skipping task ${IDX}." >&2
    exit 1
fi
echo "Resuming from: ${CKPT}"

python -u main.py \
  --env SIG \
  --loc $LOC \
  --algo $ALGO \
  --n_agent 16 \
  --seed $SEED \
  --trial_run $TRIAL \
  --env_config Configuration/env_ratios/${RATIO}.json \
  --train_data Environment/SUMOData/NFIG_k16.csv \
  --test_data Environment/SUMOData/NFIG_k16.csv \
  --resume_from "$CKPT"
