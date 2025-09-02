#!/bin/bash
# Script para activar el virtual environment correctamente

echo "🚀 Activando virtual environment..."
cd /home/jack/botijo
source venv_chatgpt/bin/activate

echo "✅ Virtual environment activado:"
echo "   Python: $(which python)"
echo "   Versión: $(python --version)"
echo ""
echo "🤖 Para ejecutar scripts:"
echo "   python servo10.py"
echo "   python integrator5.py"
echo ""
