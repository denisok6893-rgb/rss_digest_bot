#!/data/data/com.termux/files/usr/bin/bash
set -e

cd ~/rss_digest_bot || exit 1
source .venv/bin/activate

echo "VENV:" $(which python)
echo "Starting RSS Digest Bot..."
python -u app/main.py
