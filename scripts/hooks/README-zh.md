# Git Hooks

这个目录包含了项目的git hooks，用于在git操作时自动执行代码质量检查。

## 可用的Hooks

### pre-commit
在每次`git commit`之前自动执行：
- `make lint` - 代码质量检查
- `make add-license` - 自动添加Apache许可证头

## 安装方法

### 方法1：使用make命令（推荐）
```bash
make dev
```

### 方法2：直接运行安装脚本
```bash
./scripts/install-hooks.sh
```

### 方法3：手动安装
```bash
cp scripts/hooks/* .git/hooks/
chmod +x .git/hooks/*
```

## 工作流程

1. 当你运行`git commit`时，pre-commit hook会自动执行
2. 首先运行`make lint`检查代码质量
3. 然后运行`make add-license`添加许可证头
4. 如果有文件被修改（如添加了许可证头），需要重新提交
5. 如果检查失败，提交会被阻止

## 跳过Hooks（不推荐）

如果在特殊情况下需要跳过hooks，可以使用：
```bash
git commit --no-verify -m "commit message"
```

**注意：** 跳过hooks可能导致代码质量问题，请谨慎使用。

## 团队协作

- 所有团队成员在克隆仓库后都应该运行`make dev`来安装hooks
- hooks脚本存储在`scripts/hooks/`目录中，可以被git跟踪
- 对hooks的修改会影响整个团队，请谨慎修改 