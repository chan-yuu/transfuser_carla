#!/bin/bash
# =============================================================================
# TransFuser Longest6 Evaluation Script
# =============================================================================
# Usage:
#   1. Start CARLA server first:
#      cd /home/cyun/APP/carla/CARLA_0.9.10.1
#      ./CarlaUE4.sh --world-port=2000 -opengl
#
#   2. Activate conda environment and run this script:
#      conda activate tfuse
#      ./run_evaluation.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Basic Path Configuration
# -----------------------------------------------------------------------------
export CARLA_ROOT=/home/cyun/APP/carla/CARLA_0.9.10.1
export WORK_DIR=/home/cyun/Project/carla/transfuser

# -----------------------------------------------------------------------------
# PYTHONPATH Configuration (CARLA API + scenario_runner + leaderboard)
# -----------------------------------------------------------------------------
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

# -----------------------------------------------------------------------------
# Evaluation Configuration
# -----------------------------------------------------------------------------
# Scenario configuration file
export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json

# Route configuration file (36 evaluation routes)
export ROUTES=${WORK_DIR}/leaderboard/data/longest6/longest6.xml

# Number of repetitions per route
export REPETITIONS=1

# Track: SENSORS or MAP
export CHALLENGE_TRACK_CODENAME=SENSORS

# Output JSON file
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/transfuser_longest6.json

# -----------------------------------------------------------------------------
# Agent Configuration
# -----------------------------------------------------------------------------
# Agent Python file (HybridAgent class)
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/submission_agent.py

# Model weights directory (contains args.txt and .pth files)
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/transfuser

# -----------------------------------------------------------------------------
# Debug and Visualization Configuration
# -----------------------------------------------------------------------------
# DEBUG_CHALLENGE:
#   0 = no debug
#   1 = basic debug output
#   2 = verbose debug output
export DEBUG_CHALLENGE=1

# Visualization save path (enables visualization when set)
export SAVE_PATH=/home/cyun/Project/carla/transfuser/results/visualizations
mkdir -p ${SAVE_PATH}

# RESUME:
#   1 = resume from last checkpoint
export RESUME=1

# DATAGEN:
#   0 = evaluation mode
#   1 = data generation mode
export DATAGEN=0

# -----------------------------------------------------------------------------
# Run Evaluation
# -----------------------------------------------------------------------------
echo "=============================================="
echo "TransFuser Longest6 Evaluation"
echo "=============================================="
echo "CARLA_ROOT: ${CARLA_ROOT}"
echo "WORK_DIR: ${WORK_DIR}"
echo "SCENARIOS: ${SCENARIOS}"
echo "ROUTES: ${ROUTES}"
echo "TEAM_AGENT: ${TEAM_AGENT}"
echo "TEAM_CONFIG: ${TEAM_CONFIG}"
echo "SAVE_PATH: ${SAVE_PATH}"
echo "=============================================="

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
    --scenarios=${SCENARIOS} \
    --routes=${ROUTES} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME}

echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${CHECKPOINT_ENDPOINT}"
echo "Visualizations saved to: ${SAVE_PATH}"
echo "=============================================="