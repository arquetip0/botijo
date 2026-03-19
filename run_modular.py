#!/usr/bin/env python3
"""Launcher for modular src/main.py — used by panel backend."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "src"))
sys.path.insert(0, os.path.join(here, "vendor"))

from main import main
main()
