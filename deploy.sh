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
rsync -avz requirements.txt "$REMOTE:$REMOTE_DIR/" 2>/dev/null || true
echo "✅ Sync complete"

if [ "$1" = "--run" ]; then
    shift
    echo "🤖 Launching Botijo..."
    ssh -t "$REMOTE" "cd $REMOTE_DIR && source venv_chatgpt/bin/activate && PYTHONPATH=src:vendor python src/main.py $@"
fi
