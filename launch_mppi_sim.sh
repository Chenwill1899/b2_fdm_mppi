#!/bin/bash
# Launch ausim2 + Geomapping + MPPI closed-loop with automatic cleanup on Ctrl+C.

set -e

AUSIM_DIR="/home/mexxiie/prj/ausim2"
GEO_DIR="/home/mexxiie/prj/Geomapping_ros2"
PYMPP_DIR="/home/mexxiie/prj/py-mppi"
LOG_DIR="${PYMPP_DIR}/logs"
mkdir -p "${LOG_DIR}"

AUSIM_PID=""
GEO_PID=""
MPPI_PID=""

cleanup_all() {
    echo ""
    echo "[cleanup] Stopping all processes..."
    [ -n "${MPPI_PID}" ] && kill -9 "${MPPI_PID}" 2>/dev/null || true
    [ -n "${GEO_PID}" ] && kill -9 "${GEO_PID}" 2>/dev/null || true
    pkill -9 -P "${GEO_PID}" 2>/dev/null || true
    [ -n "${AUSIM_PID}" ] && kill -9 "${AUSIM_PID}" 2>/dev/null || true
    pkill -9 -f "scout --headless" 2>/dev/null || true
    pkill -9 -f "ausim_ros_bridge" 2>/dev/null || true
    pkill -9 -f "traversability_map" 2>/dev/null || true
    pkill -9 -f "traversability_filter" 2>/dev/null || true
    pkill -9 -f "traversability_cost" 2>/dev/null || true
    pkill -9 -f "terrain_pub_node" 2>/dev/null || true
    pkill -9 -f "MEDIRL" 2>/dev/null || true
    pkill -9 -f "static_transform_publisher" 2>/dev/null || true
    echo "[cleanup] Done."
}

# Pre-launch cleanup: ensure no zombie processes from previous runs
echo "[0/3] Cleaning up previous processes..."
pkill -9 -f "scout --headless" 2>/dev/null || true
pkill -9 -f "ausim_ros_bridge" 2>/dev/null || true
sleep 2

trap cleanup_all INT TERM EXIT

echo "[1/3] Starting ausim2 MuJoCo (headless)..."
cd "${AUSIM_DIR}"
./em_run.sh --headless > "${LOG_DIR}/ausim2.log" 2>&1 &
AUSIM_PID=$!
sleep 8
echo "[1/3] ausim2 PID: ${AUSIM_PID}"

# Verify robot is at origin before proceeding
python3 -c "
import rclpy, time
from nav_msgs.msg import Odometry
rclpy.init()
node = rclpy.create_node('origin_check')
msg = None
def cb(m):
    global msg
    msg = m
sub = node.create_subscription(Odometry, '/scout1/odom', cb, 10)
for _ in range(50):
    rclpy.spin_once(node, timeout_sec=0.1)
    if msg:
        p = msg.pose.pose.position
        dist = (p.x - 1.0)**2 + p.y**2
        if dist < 0.01:
            print(f'[1/3] Robot at origin: ({p.x:.3f}, {p.y:.3f})')
        else:
            print(f'[1/3] WARNING: Robot not at origin: ({p.x:.3f}, {p.y:.3f})')
        break
node.destroy_node()
rclpy.shutdown()
"

echo "[2/3] Starting Geomapping..."
cd "${GEO_DIR}"
source install/setup.bash
ros2 launch traversability_mapping ausim_cube_mppi.launch.py launch_rviz:=false use_medirl:=true > "${LOG_DIR}/geomapping.log" 2>&1 &
GEO_PID=$!
sleep 8
echo "[2/3] Geomapping PID: ${GEO_PID}"

echo "[3/3] Starting MPPI closed-loop..."
cd "${PYMPP_DIR}"
export LD_LIBRARY_PATH="${GEO_DIR}/install/elevation_msgs/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${GEO_DIR}/install/elevation_msgs/local/lib/python3.10/dist-packages:${PYTHONPATH}"
/usr/bin/python3 tools/fdm_mppi.py mujoco-closed-loop \
    --profile configs/mujoco_test_obstacles_localmap.yaml \
    --controller nominal_numpy > "${LOG_DIR}/mppi.log" 2>&1 &
MPPI_PID=$!
echo "[3/3] MPPI PID: ${MPPI_PID}"

echo ""
echo "All processes started. Press Ctrl+C to stop everything."
echo "Logs: ${LOG_DIR}/"

wait "${MPPI_PID}" 2>/dev/null || true

trap - INT TERM EXIT
cleanup_all
