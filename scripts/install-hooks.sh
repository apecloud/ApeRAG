#!/bin/bash

# Install git hooks script
# This script copies git hooks from scripts/hooks/ to .git/hooks/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_SOURCE_DIR="$SCRIPT_DIR/hooks"
HOOKS_TARGET_DIR="$PROJECT_ROOT/.git/hooks"

echo "Installing git hooks..."

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Create hooks source directory if it doesn't exist
mkdir -p "$HOOKS_SOURCE_DIR"

# Copy hooks from source to .git/hooks/
if [ -d "$HOOKS_SOURCE_DIR" ]; then
    for hook in "$HOOKS_SOURCE_DIR"/*; do
        if [ -f "$hook" ]; then
            hook_name=$(basename "$hook")
            echo "Installing $hook_name hook..."
            cp "$hook" "$HOOKS_TARGET_DIR/$hook_name"
            chmod +x "$HOOKS_TARGET_DIR/$hook_name"
            echo "✅ $hook_name hook installed"
        fi
    done
else
    echo "❌ Error: Hooks source directory not found: $HOOKS_SOURCE_DIR"
    exit 1
fi

echo "🎉 All git hooks installed successfully!"
echo "📝 Note: Run 'make dev' or 'scripts/install-hooks.sh' after cloning the repository to install hooks." 