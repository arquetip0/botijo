#!/bin/bash
# Deploy Botijo code to Raspberry Pi
# Usage: ./deploy.sh [--run [--mode MODE]]

set -e

REMOTE="botijo"
REMOTE_DIR="~/botijo"

echo "📦 Syncing code to $REMOTE..."
rsync -avz --delete src/ "$REMOTE:$REMOTE_DIR/src/"
rsync -avz --delete config/ "$REMOTE:$REMOTE_DIR/config/"
rsync -avz --delete tools/ "$REMOTE:$REMOTE_DIR/tools/"
rsync -avz --delete vendor/ "$REMOTE:$REMOTE_DIR/vendor/"
rsync -avz --delete --copy-links old/ "$REMOTE:$REMOTE_DIR/old/"
rsync -avz requirements.txt "$REMOTE:$REMOTE_DIR/" 2>/dev/null || true
rsync -avz run_livekit.py "$REMOTE:$REMOTE_DIR/"

# Panel web (lives outside ~/botijo/ on RPi)
rsync -avz --delete panel/ "$REMOTE:~/panel/"
rsync -avz panel_backend/app.py "$REMOTE:~/panel_backend/app.py"
ssh "$REMOTE" "sudo systemctl restart panel-backend"
echo "✅ Sync complete"

# Note: first deploy with LiveKit needs: pip install livekit livekit-api sounddevice

if [ "$1" = "--run" ]; then
    shift
    echo "🤖 Launching Botijo..."
    ssh -t "$REMOTE" "cd $REMOTE_DIR && source venv_chatgpt/bin/activate && PYTHONPATH=src:vendor python src/main.py $@"
fi
