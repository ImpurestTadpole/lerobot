#!/usr/bin/env bash
# Preflight + smoke test for XLerobot (see Guide.sh TELEOPERATION section).
# Run on the Jetson with motors/cameras/VR connected.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
info() { echo -e "[INFO] $*"; }

RUN_TELEOP=false
TELEOP_SECONDS=0
for arg in "$@"; do
    case "$arg" in
        --teleop) RUN_TELEOP=true ;;
        --teleop-seconds=*) TELEOP_SECONDS="${arg#*=}" ;;
        -h|--help)
            cat <<'EOF'
Usage: scripts/test_xlerobot.sh [--teleop] [--teleop-seconds=N]

Preflight checks for XLerobot (USB, cameras, VR, Python imports).

  --teleop              After checks, run lerobot-teleoperate (Ctrl+C to stop)
  --teleop-seconds=N    Run teleop for N seconds then exit (requires `timeout`)

Guide.sh reference: TELEOPERATION + CAMERA SETUP sections.
EOF
            exit 0
            ;;
    esac
done

# --- Environment ---
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "lerobot" ]]; then
    if [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/miniconda/etc/profile.d/conda.sh"
        conda activate lerobot
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        conda activate lerobot
    else
        warn "conda env 'lerobot' not active; continuing with current python"
    fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
info "Repo: $REPO_ROOT"

# --- Software smoke test ---
python - <<'PY' || fail "Python import smoke test failed"
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from lerobot.teleoperators.xlerobot_vr.configuration_xlerobot_vr import XLerobotVRTeleopConfig
from lerobot.teleoperators.xlerobot_vr.xlerobot_vr import VR_AVAILABLE

cfg = XLerobotConfig()
assert cfg.type == "xlerobot"
assert set(cfg.cameras) >= {"head", "left_wrist", "right_wrist"}
assert XLerobotVRTeleopConfig().type == "xlerobot_vr"
print(f"xlerobot ports: {cfg.port1}, {cfg.port2}")
print(f"VR_AVAILABLE: {VR_AVAILABLE}")
PY
pass "Python imports (xlerobot + xlerobot_vr)"

command -v lerobot-teleoperate >/dev/null 2>&1 && pass "lerobot-teleoperate CLI" || fail "lerobot-teleoperate not on PATH"

# --- USB motor ports (Guide.sh: sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1) ---
MISSING_PORTS=()
for dev in /dev/ttyACM0 /dev/ttyACM1; do
    if [[ -e "$dev" ]]; then
        perms=$(stat -c "%a" "$dev" 2>/dev/null || echo "?")
        pass "Motor port $dev (mode $perms)"
        if [[ "$perms" != "666" ]] && [[ ! -w "$dev" ]]; then
            warn "$dev not writable — run: sudo chmod 666 $dev"
        fi
    else
        MISSING_PORTS+=("$dev")
    fi
done
if ((${#MISSING_PORTS[@]} > 0)); then
    warn "Missing motor ports: ${MISSING_PORTS[*]} (robot USB not connected?)"
fi

# --- Cameras ---
if command -v v4l2-ctl >/dev/null 2>&1; then
    for dev in /dev/video0 /dev/video4 /dev/video6 /dev/video8; do
        if [[ -e "$dev" ]]; then
            fmt=$(v4l2-ctl -d "$dev" --get-fmt-video 2>/dev/null | grep -i "Pixel Format" || true)
            if [[ -n "$fmt" ]]; then
                if echo "$fmt" | grep -qi MJPG; then
                    pass "Camera $dev: MJPG"
                else
                    warn "Camera $dev: $fmt (expected MJPG for wrist cams — see Guide.sh CAMERA SETUP)"
                fi
            else
                pass "Camera $dev present"
            fi
        fi
    done
else
    warn "v4l2-ctl not installed; skipping V4L2 format check"
fi

if command -v lerobot-find-cameras >/dev/null 2>&1; then
    info "Detected cameras:"
    lerobot-find-cameras opencv 2>/dev/null | head -20 || warn "lerobot-find-cameras opencv failed"
fi

# RealSense head camera (serial in config_xlerobot.py)
python - <<'PY' 2>/dev/null || warn "RealSense head camera not detected (pyrealsense2 or device missing)"
import pyrealsense2 as rs
ctx = rs.context()
devs = ctx.query_devices()
serial = "342222071125"
found = any(d.get_info(rs.camera_info.serial_number) == serial for d in devs)
print(f"RealSense devices: {devs.size()}")
if found:
    print(f"Head camera serial {serial}: OK")
else:
    print(f"Head camera serial {serial}: NOT FOUND")
PY

# --- VR (XLeVR path from configuration_xlerobot_vr.py) ---
VR_PATH=$(python - <<'PY'
from lerobot.teleoperators.xlerobot_vr.configuration_xlerobot_vr import XLerobotVRTeleopConfig
print(XLerobotVRTeleopConfig().xlevr_path or "")
PY
)
if [[ -n "$VR_PATH" ]] && [[ -d "$VR_PATH" ]]; then
    pass "XLeVR path: $VR_PATH"
else
    warn "XLeVR path missing or not a directory: ${VR_PATH:-<unset>}"
    warn "Set --teleop.xlevr_path=... or edit configuration_xlerobot_vr.py"
fi

# --- GPU (optional on Jetson) ---
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true
fi

echo ""
info "Preflight complete."

if [[ "$RUN_TELEOP" != true ]]; then
    cat <<'EOF'

Next: VR teleop smoke test (Guide.sh):
  lerobot-teleoperate \
      --robot.type=xlerobot \
      --teleop.type=xlerobot_vr \
      --display_data=true

Remote Rerun viewer (on your laptop, replace JETSON_IP):
  rerun --serve-web --web-viewer-port 9090 --connect "rerun+http://JETSON_IP:9876/proxy"

Re-run with --teleop to start teleoperation after checks.
EOF
    exit 0
fi

info "Starting lerobot-teleoperate (Ctrl+C to stop)..."
TELEOP_CMD=(
    lerobot-teleoperate
    --robot.type=xlerobot
    --teleop.type=xlerobot_vr
    --display_data=true
)

if [[ "$TELEOP_SECONDS" -gt 0 ]]; then
    timeout "$TELEOP_SECONDS" "${TELEOP_CMD[@]}"
else
    "${TELEOP_CMD[@]}"
fi
