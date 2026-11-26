#!/usr/bin/env bash
#SBATCH --output=local_data/experiments/logs/out/slurm-%j.out
#SBATCH --error=local_data/experiments/logs/err/slurm-%j.err

# Example of usage:
# sbatch --gres=gpu:1 launcher.sh --backbone CLIP-ViTB16 --procedure scp --K 16 --solver gd --alpha_icp 0.05 --alpha_fcp 0.0 --seeds 20

# Default values
backbone="CLIP-ViTB16"
seeds=20
K=16
solver="lda"
alpha_icp=0.0
alpha_fcp=0.0
procedure="icp"

# Parse long options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone)
      backbone="$2"
      shift 2
      ;;
    --seeds)
      seeds="$2"
      shift 2
      ;;
    --K)
      K="$2"
      shift 2
      ;;
    --solver)
      solver="$2"
      shift 2
      ;;
    --alpha_icp)
      alpha_icp="$2"
      shift 2
      ;;
    --alpha_fcp)
      alpha_fcp="$2"
      shift 2
      ;;
    --procedure)
      procedure="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--backbone B] [--K N] [--solver S] [--alpha_icp X] [--alpha_fcp X] [--procedure P] [--seeds N]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Activate environment
eval "$(conda shell.bash hook)"
conda activate fewshotclip

# Run script
PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH}" \
python run_conformal.py \
  --backbone "$backbone" \
  --K "$K" \
  --solver "$solver" \
  --alpha_icp "$alpha_icp" \
  --alpha_fcp "$alpha_fcp" \
  --procedure "$procedure" \
  --seeds "$seeds"