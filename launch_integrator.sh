#!/bin/bash
# filepath: /home/jack/botijo/launch_integrator.sh

# Change to script directory
cd /home/jack/botijo

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set Python path to include venv packages
export PYTHONPATH="/home/jack/botijo/venv_chatgpt/lib/python3.11/site-packages:$PYTHONPATH"

# Run the script with system Python
exec /usr/bin/python3 integrator3.py