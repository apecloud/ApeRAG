#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from pathlib import Path

def download_from_modelscope(model_name: str, save_dir: str):
    """Download model from modelscope"""
    try:
        from modelscope import snapshot_download
        print(f"Downloading {model_name} from modelscope...")
        model_dir = snapshot_download(model_name, cache_dir=save_dir)
        print(f"Model downloaded to: {model_dir}")
        return model_dir
    except ImportError:
        print("Error: modelscope is not installed. Please install it with: pip install modelscope")
        sys.exit(1)

def download_from_huggingface(model_name: str, save_dir: str):
    """Download model from huggingface"""
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_name} from huggingface...")
        model_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=save_dir,
            local_files_only=False,
            resume_download=True
        )
        print(f"Model downloaded to: {model_dir}")
        return model_dir
    except ImportError:
        print("Error: huggingface_hub is not installed. Please install it with: pip install huggingface_hub")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Download models from modelscope or huggingface')
    parser.add_argument('--source', choices=['modelscope', 'huggingface'], required=True,
                      help='Source to download from (modelscope or huggingface)')
    parser.add_argument('--model', type=str, required=True,
                      help='Model name to download')
    parser.add_argument('--save-dir', type=str, required=True,
                      help='Directory to save the model')
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.source == 'modelscope':
        download_from_modelscope(args.model, args.save_dir)
    else:
        download_from_huggingface(args.model, args.save_dir)

if __name__ == '__main__':
    main() 