#!/bin/bash

# Install git hooks script
# This script configures tracked hooks via core.hooksPath so they are easier to
# keep consistent across local clones and worktrees.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_SOURCE_DIR="$SCRIPT_DIR/hooks"
echo "Installing git hooks..."

# Check if we're in a git repository
if ! git -C "$PROJECT_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

if [ ! -d "$HOOKS_SOURCE_DIR" ]; then
    echo "❌ Error: Hooks source directory not found: $HOOKS_SOURCE_DIR"
    exit 1
fi

for hook in "$HOOKS_SOURCE_DIR"/*; do
    if [ -f "$hook" ]; then
        hook_name="$(basename "$hook")"
        case "$hook_name" in
            README*)
                continue
                ;;
        esac
        chmod +x "$hook"
    fi
done

echo "Configuring core.hooksPath -> scripts/hooks"
git -C "$PROJECT_ROOT" config core.hooksPath scripts/hooks

HOOKS_PATH="$(git -C "$PROJECT_ROOT" config --get core.hooksPath)"
echo "✅ Git hooks enabled via core.hooksPath=$HOOKS_PATH"
echo "🎉 All git hooks installed successfully!"
echo "📝 Note: Run 'make dev', 'make env-dev', or './scripts/install-hooks.sh' after cloning a fresh repository to enable hooks."
