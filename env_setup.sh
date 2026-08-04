#!/usr/bin/env bash
# ==============================================================================
# SHARP++ Environment Setup Script
# Usage: source env_setup.sh
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate Virtual Environment
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "[+] Virtual environment activated: $VIRTUAL_ENV"
else
    echo "[!] Virtual environment not found at $SCRIPT_DIR/.venv"
fi

# Load Environment Variables from .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
    echo "[+] Environment variables loaded from .env"
fi

# Set PYTHONPATH to project root
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH}"
echo "[+] PYTHONPATH configured: $PYTHONPATH"
