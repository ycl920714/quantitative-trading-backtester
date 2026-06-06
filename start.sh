#!/bin/bash
# ════════════════════════════════════════════════
# Investment Assistant — Mac Startup Script
# Run this once to set everything up
# ════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "  🤖 Investment Assistant Setup"
echo "  ════════════════════════════"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found. Install from https://www.python.org"
    exit 1
fi

echo "✅ Python: $(python3 --version)"

# Check .env
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo ""
    echo "📝 Created .env file from template."
    echo "   ⚠️  Please edit .env with your API keys before continuing!"
    echo "   Run: open -e $SCRIPT_DIR/.env"
    echo ""
    echo "   Then run this script again."
    exit 0
fi

# Install dependencies
echo "📦 Installing Python packages..."
pip3 install -r "$SCRIPT_DIR/requirements.txt" --quiet

echo ""
echo "🚀 Starting Investment Assistant..."
echo "   Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
python3 investment_assistant.py
