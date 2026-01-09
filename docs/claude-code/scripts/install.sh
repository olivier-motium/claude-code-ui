#!/bin/bash
#
# Claude Code Toolkit Installer
#
# This script backs up your existing ~/.claude config and creates
# symlinks to this repository.
#
# Usage:
#   ./scripts/install.sh          # Interactive install
#   ./scripts/install.sh --force  # Skip confirmation
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$REPO_DIR/config"

echo -e "${GREEN}Claude Code Toolkit Installer${NC}"
echo "==============================="
echo ""
echo "This will install Claude Code Toolkit by creating symlinks"
echo "from ~/.claude/ to $CONFIG_DIR"
echo ""

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo -e "${RED}Error: Config directory not found at $CONFIG_DIR${NC}"
    exit 1
fi

# Check for --force flag
FORCE=false
if [ "$1" = "--force" ]; then
    FORCE=true
fi

# Confirm unless --force
if [ "$FORCE" = false ]; then
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Backup existing config
if [ -d "$HOME/.claude" ] || [ -L "$HOME/.claude" ]; then
    BACKUP_DIR="$HOME/.claude.backup.$(date +%s)"
    echo -e "${YELLOW}Backing up existing ~/.claude to $BACKUP_DIR${NC}"
    mv "$HOME/.claude" "$BACKUP_DIR"
fi

# Create ~/.claude directory
echo "Creating ~/.claude directory..."
mkdir -p "$HOME/.claude"

# Create symlinks
echo "Creating symlinks..."

# Settings file
if [ -f "$CONFIG_DIR/settings.json" ]; then
    ln -sf "$CONFIG_DIR/settings.json" "$HOME/.claude/settings.json"
    echo "  ✓ settings.json"
fi

# Commands directory
if [ -d "$CONFIG_DIR/commands" ]; then
    ln -sf "$CONFIG_DIR/commands" "$HOME/.claude/commands"
    echo "  ✓ commands/"
fi

# Hooks directory
if [ -d "$CONFIG_DIR/hooks" ]; then
    ln -sf "$CONFIG_DIR/hooks" "$HOME/.claude/hooks"
    echo "  ✓ hooks/"
fi

# Skills directory
if [ -d "$CONFIG_DIR/skills" ]; then
    ln -sf "$CONFIG_DIR/skills" "$HOME/.claude/skills"
    echo "  ✓ skills/"
fi

# Make hooks executable
echo "Making hooks executable..."
if [ -d "$CONFIG_DIR/hooks" ]; then
    chmod +x "$CONFIG_DIR/hooks"/*.py 2>/dev/null || true
    echo "  ✓ hooks/*.py"
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Start Claude Code: claude"
echo "  2. You should see: 'SessionStart:startup hook success: MANDATORY...'"
echo "  3. Try a command: /QA"
echo ""
echo "To uninstall:"
echo "  rm -rf ~/.claude"
if [ -n "$BACKUP_DIR" ]; then
    echo "  mv $BACKUP_DIR ~/.claude  # restore backup"
fi
