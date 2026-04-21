#!/usr/bin/env bash
# deploy/install.sh — One-liner VPS setup for poly-v2
#
# Usage (on a fresh Ubuntu 22.04 VPS):
#   curl -fsSL https://raw.githubusercontent.com/Wenbot-club/poly-v2/main/deploy/install.sh | bash
#
# What it does:
#   1. Installs system dependencies (python3.11, git, pip)
#   2. Clones the repo to /opt/poly-v2
#   3. Creates a virtualenv and installs Python deps
#   4. Runs the interactive setup wizard (collect credentials)
#   5. Runs the bootstrap check (geoblock + CLOB + approvals)
#   6. Installs and starts the systemd service (paper-live mode)

set -euo pipefail

REPO_URL="https://github.com/Wenbot-club/poly-v2.git"
INSTALL_DIR="/opt/poly-v2"
LOG_DIR="/var/log/poly-v2"
VENV="$INSTALL_DIR/.venv"
ENV_FILE="$INSTALL_DIR/.env"

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   poly-v2 — VPS installer                       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "→ Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl
ok "System dependencies installed"

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "→ Repo already cloned — pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "→ Cloning repo to $INSTALL_DIR..."
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown "$(whoami):$(whoami)" "$INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi
ok "Repo ready at $INSTALL_DIR"

# ── 3. Virtualenv + deps ──────────────────────────────────────────────────────
echo "→ Creating virtualenv..."
python3.11 -m venv "$VENV"
echo "→ Installing Python dependencies..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"
ok "Python environment ready"

# ── 4. Log directory ──────────────────────────────────────────────────────────
sudo mkdir -p "$LOG_DIR"
sudo chown "$(whoami):$(whoami)" "$LOG_DIR"
ok "Log directory ready at $LOG_DIR"

# ── 5. Wizard ─────────────────────────────────────────────────────────────────
echo ""
echo "→ Running setup wizard..."
"$VENV/bin/python" "$INSTALL_DIR/scripts/setup_wizard.py" --env-file "$ENV_FILE"

# ── 6. Bootstrap check ────────────────────────────────────────────────────────
echo ""
echo "→ Running bootstrap check..."
set -a; source "$ENV_FILE"; set +a
if "$VENV/bin/python" "$INSTALL_DIR/demos/bootstrap_check.py"; then
    ok "Bootstrap check passed — GO"
else
    die "Bootstrap check failed — fix the issues above before continuing"
fi

# ── 7. Systemd service ────────────────────────────────────────────────────────
echo ""
echo "→ Installing systemd service..."

# Patch service file with env vars and current user
CURRENT_USER="$(whoami)"
sudo tee /etc/systemd/system/btc_m5.service > /dev/null <<SERVICE
[Unit]
Description=BTC M5 paper-live runner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV/bin/python $INSTALL_DIR/demos/demo_btc_m5_live.py \\
    --windows 2016 \\
    --output-dir $INSTALL_DIR/m5_out_live

Restart=on-failure
RestartSec=30
StandardOutput=append:$LOG_DIR/btc_m5.log
StandardError=append:$LOG_DIR/btc_m5.log
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable btc_m5
sudo systemctl start btc_m5
ok "Service btc_m5 started"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Installation complete — bot running            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Logs  : tail -f $LOG_DIR/btc_m5.log"
echo "  Status: sudo systemctl status btc_m5"
echo "  Stop  : sudo systemctl stop btc_m5"
echo ""
