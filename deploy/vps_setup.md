# BTC M5 VPS deployment guide

Target: Ubuntu 22.04 LTS, 1 vCPU / 1 GB RAM minimum.

## 1. Install system dependencies

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
```

## 2. Clone and install

```bash
sudo mkdir -p /opt/poly-v2
sudo chown ubuntu:ubuntu /opt/poly-v2
git clone https://github.com/Wenbot-club/poly-v2 /opt/poly-v2
cd /opt/poly-v2
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 3. Prepare log directory

```bash
sudo mkdir -p /var/log/poly-v2
sudo chown ubuntu:ubuntu /var/log/poly-v2
```

## 4. Install and start the systemd service

```bash
sudo cp deploy/btc_m5.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable btc_m5
sudo systemctl start btc_m5
```

## 5. Check status and logs

```bash
# Status
sudo systemctl status btc_m5

# Live logs
tail -f /var/log/poly-v2/btc_m5.log

# Last 100 lines
tail -n 100 /var/log/poly-v2/btc_m5.log
```

## 6. Restart after a code update

```bash
cd /opt/poly-v2
git pull
sudo systemctl restart btc_m5
```

## 7. Stop

```bash
sudo systemctl stop btc_m5
```

## 8. Output artifacts

Per-run artifacts land in `/opt/poly-v2/m5_out_live/`:

| File | Contents |
|------|----------|
| `window_NNN.json` | Per-window TradeRecord |
| `m5_campaign_summary.json` | Aggregated campaign stats |
| `latency_summary.json` | Per-order latency records + summary stats |

## Notes

- The service runs `--windows 2016` (~7 days of 5-min windows). Adjust as needed.
- On restart, a new output dir is overwritten. To preserve history, use `--output-dir` with a timestamped path or change the service file.
- No credentials or API keys are needed for paper-live mode.
