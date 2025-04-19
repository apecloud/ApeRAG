# ApeRAG 项目 PyCharm 调试指南

本文详细介绍如何在 PyCharm 中调试 ApeRAG 项目的两个核心模块：Celery 异步任务服务和 Web 后端服务。

---

## 一、调试 Celery 异步任务

### 1.1 创建调试配置
1. 点击 PyCharm 顶部菜单 **Run > Edit Configurations**
2. 点击 **+** 选择 **Python** 配置类型
3. 按以下参数配置：

**核心配置项：**
```python
名称: celery
Python解释器: [通过 uv 虚拟环境获取]
  # 获取方法：终端运行 `readlink .venv/bin/python`
脚本路径: [celery可执行文件路径]
  # 获取方法：终端运行 `which celery`
参数: -A config.celery worker -l INFO --pool=solo
  # 关键参数：必须包含--pool=solo（单进程模式便于调试）
环境变量: 
  PYTHONUNBUFFERED=1;PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

![celery.jpeg](images%2Fcelery.jpeg)

---

## 二、调试 Web 后端服务

### 2.1 创建调试配置
1. 点击 **Run > Edit Configurations**
2. 点击 **+** 选择 **Python** 配置类型
3. 按以下参数配置：

**核心配置项：**
```python
名称: backend
Python解释器: [与Celery相同的uv虚拟环境]
脚本路径: [uvicorn可执行文件路径]
  # 获取方法：which uvicorn
参数: config.asgi:application --host 0.0.0.0 --port 8000 --reload --reload-include '*.html'
环境变量:
  PYTHONUNBUFFERED=1;
  DJANGO_SETTINGS_MODULE=config.settings;
```

![backend.jpeg](images%2Fbackend.jpeg)
