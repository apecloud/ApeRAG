# Git Hooks

This directory contains the project's Git hooks, which are used to automatically perform code quality checks during Git operations.

## Available Hooks

### pre-commit
Automatically executes before every `git commit`:
- `make lint` - Performs code quality checks.
- `make add-license` - Automatically adds the Apache license header.

---
## Installation Methods

### Method 1: Using `make` (Recommended)
```bash
make dev
```

### Method 2: Directly Run the Installation Script
```bash
./scripts/install-hooks.sh
```

### Method 3: Manual Installation
```bash
cp scripts/hooks/* .git/hooks/
chmod +x .git/hooks/*
```

---
## Workflow

1.  When you run `git commit`, the pre-commit hook automatically executes.
2.  First, `make lint` runs to check code quality.
3.  Then, `make add-license` runs to add the license header.
4.  If files are modified (e.g., by adding the license header), you'll need to re-commit.
5.  If checks fail, the commit will be blocked.

---
## Skipping Hooks (Not Recommended)

In special circumstances, if you need to skip the hooks, you can use:
```bash
git commit --no-verify -m "commit message"
```

**Note:** Skipping hooks can lead to code quality issues. Please use with caution.

---
## Team Collaboration

-   All team members should run `make dev` to install the hooks after cloning the repository.
-   Hook scripts are stored in the `scripts/hooks/` directory and can be tracked by Git.
-   Modifications to hooks will affect the entire team. Please make changes carefully.
Got it. Here's the English version of the Git Hooks documentation, keeping it concise as you prefer:

---
# Git Hooks

This directory contains the project's Git hooks, which automatically run code quality checks during Git operations.

---
## Available Hooks

### `pre-commit`
Automatically runs before every `git commit`:
- `make lint` - performs code quality checks.
- `make add-license` - automatically adds the Apache license header.

---
## Installation

### Method 1: Using `make` (Recommended)
```bash
make dev
```

### Method 2: Running the Install Script Directly
```bash
./scripts/install-hooks.sh
```

### Method 3: Manual Installation
```bash
cp scripts/hooks/* .git/hooks/
chmod +x .git/hooks/*
```

---
## Workflow

1.  When you run `git commit`, the `pre-commit` hook automatically executes.
2.  First, `make lint` checks code quality.
3.  Then, `make add-license` adds the license header.
4.  If files are modified (e.g., by adding the license header), a re-commit is required.
5.  If checks fail, the commit is prevented.

---
## Skipping Hooks (Not Recommended)

To skip hooks in special cases, use:
```bash
git commit --no-verify -m "commit message"
```

**Note:** Skipping hooks can lead to code quality issues. Use with caution.

---
## Team Collaboration

* All team members should run `make dev` after cloning the repository to install the hooks.
* Hook scripts are stored in the `scripts/hooks/` directory and are tracked by Git.
* Changes to hooks affect the entire team, so modify them carefully.