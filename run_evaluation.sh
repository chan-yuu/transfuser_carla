#!/bin/bash
# =============================================================================
# TransFuser Longest6 Evaluation Script
# =============================================================================
# Usage:
#   ./run_evaluation.sh
#
# This script will:
#   1. Activate conda environment
#   2. Kill any existing CARLA processes
#   3. Start CARLA server automatically
#   4. Run the evaluation
#   5. Clean up CARLA processes on exit
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Conda Environment Activation
# -----------------------------------------------------------------------------
# Get conda root path
CONDA_ROOT=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate tfuse

# -----------------------------------------------------------------------------
# Cleanup Function - Kill all CARLA processes
# -----------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "=============================================="
    echo "Cleaning up CARLA processes..."
    echo "=============================================="
    pkill -9 -f CarlaUE4 2>/dev/null || true
    sleep 1
    echo "Cleanup completed."
}

# Register cleanup on script exit (normal or interrupted)
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Basic Path Configuration
# -----------------------------------------------------------------------------
export CARLA_ROOT=/home/cyun/APP/carla/CARLA_0.9.10.1
export WORK_DIR=/home/cyun/Project/carla/transfuser

# -----------------------------------------------------------------------------
# PYTHONPATH Configuration (CARLA API + scenario_runner + leaderboard)
# -----------------------------------------------------------------------------
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

# -----------------------------------------------------------------------------
# Evaluation Configuration
# -----------------------------------------------------------------------------
export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/transfuser_longest6.json

# -----------------------------------------------------------------------------
# Agent Configuration
# -----------------------------------------------------------------------------
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/submission_agent.py
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/transfuser

# -----------------------------------------------------------------------------
# Debug and Visualization Configuration
# -----------------------------------------------------------------------------
# 可视化
# export DEBUG_CHALLENGE=1
# export SAVE_PATH=/home/cyun/Project/carla/transfuser/results/visualizations
# 不可视化
export DEBUG_CHALLENGE=0

export RESUME=1
export DATAGEN=0

# -----------------------------------------------------------------------------
# CARLA Server Configuration
# -----------------------------------------------------------------------------
CARLA_PORT=2000
CARLA_TIMEOUT=60  # seconds to wait for CARLA to start

# -----------------------------------------------------------------------------
# Step 1: Cleanup any existing CARLA processes
# -----------------------------------------------------------------------------
echo "=============================================="
echo "TransFuser Longest6 Evaluation"
echo "=============================================="
echo ""
echo "[Step 1] Cleaning up any existing CARLA processes..."
pkill -9 -f CarlaUE4 2>/dev/null || true
sleep 1

# -----------------------------------------------------------------------------
# Step 2: Start CARLA Server
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2] Starting CARLA server on port ${CARLA_PORT}..."
echo "          This may take 15-30 seconds..."

cd ${CARLA_ROOT}
./CarlaUE4.sh --world-port=${CARLA_PORT} -opengl &
CARLA_PID=$!
cd ${WORK_DIR}

# Wait for CARLA to be ready using netcat (check if port is listening)
echo "Waiting for CARLA to be ready..."
START_TIME=$(date +%s)
CARLA_READY=0

while true; do
    # Check if CARLA process is still running
    if ! kill -0 ${CARLA_PID} 2>/dev/null; then
        echo "ERROR: CARLA process died unexpectedly!"
        exit 1
    fi

    # Check if port 2000 is open using bash /dev/tcp
    if (echo > /dev/tcp/localhost/${CARLA_PORT}) 2>/dev/null; then
        # Port is open, wait a bit more for CARLA to fully initialize
        sleep 3
        CARLA_READY=1
        break
    fi

    # Check timeout
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ ${ELAPSED} -ge ${CARLA_TIMEOUT} ]; then
        echo "ERROR: Timeout waiting for CARLA to start (${CARLA_TIMEOUT}s)"
        exit 1
    fi

    sleep 2
done

echo "CARLA server is ready! (PID: ${CARLA_PID})"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Display Configuration
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Configuration:"
echo "=============================================="
echo "CARLA_ROOT: ${CARLA_ROOT}"
echo "WORK_DIR: ${WORK_DIR}"
echo "SCENARIOS: ${SCENARIOS}"
echo "ROUTES: ${ROUTES}"
echo "TEAM_AGENT: ${TEAM_AGENT}"
echo "TEAM_CONFIG: ${TEAM_CONFIG}"
echo "SAVE_PATH: ${SAVE_PATH}"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Step 4: Run Evaluation
# -----------------------------------------------------------------------------
echo "[Step 3] Running evaluation..."
echo ""

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

# -----------------------------------------------------------------------------
# Step 5: Completion (cleanup will be called automatically via trap)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: ${CHECKPOINT_ENDPOINT}"
echo "Visualizations saved to: ${SAVE_PATH}"
echo "=============================================="