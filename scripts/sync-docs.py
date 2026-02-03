#!/usr/bin/env python3
"""
Documentation Sync Tool

Syncs documentation from docs/ to web/docs/, automatically handling image references.

Usage:
    make docs
    
    Or directly:
    python3 scripts/sync-docs.py

How It Works:
    1. Whitelist-based Sync: Only files listed in SYNC_WHITELIST are synced
    2. Automatic Image Detection: Scans markdown files for image references
    3. Smart Image Sync: Copies referenced images maintaining directory structure
    4. Timestamp Check: Only copies images if source is newer than destination

Features:
    - Markdown Sync: Copies markdown files from docs/ to web/docs/
    - Image Detection: Automatically finds images (![](path) and <img src="">)
    - Directory Structure: Maintains the same directory structure
    - Smart Copy: Only copies when source is newer (timestamp check)
    - Safe: Never modifies source files in docs/
    - URL Aware: Skips external URLs (http://, https://)

Adding More Documents:
    Edit SYNC_WHITELIST below to add more documents:
    
    SYNC_WHITELIST = [
        "en-US/design/architecture.md",
        "en-US/design/document_upload_design.md",  # Uncomment to sync
        "zh-CN/design/architecture.md",
        "zh-CN/design/document_upload_design.md",  # Uncomment to sync
    ]

Important Notes:
    - docs/ is the source of truth - always edit files there
    - web/docs/ is generated - don't edit directly
    - Run 'make docs' after editing any documentation
    - New documents must be added to SYNC_WHITELIST to be synced

File Structure:
    docs/                          # Source of truth
    ├── en-US/
    │   ├── design/
    │   │   └── architecture.md    # Source file
    │   └── images/                # Source images
    │       └── diagram.png
    └── zh-CN/
        └── design/
            └── architecture.md

    web/docs/                      # Generated (don't edit directly)
    ├── en-US/
    │   ├── design/
    │   │   └── architecture.md    # Synced file
    │   └── images/                # Synced images
    │       └── diagram.png
    └── zh-CN/
        └── design/
            └── architecture.md
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Set

# Configuration
DOCS_ROOT = Path("docs")
WEB_DOCS_ROOT = Path("web/docs")

# Whitelist: files to sync from docs/ to web/docs/
# Format: relative path from docs/ directory
SYNC_WHITELIST = [
    # English docs - Design
    "en-US/design/architecture.md",
    "en-US/design/document_upload_design.md",
    "en-US/design/graph_index_creation.md",
    # "en-US/design/chat_history_design.md",
    
    # English docs - Development
    "en-US/development/development-guide.md",
    
    # English docs - Deployment
    "en-US/deployment/build-docker-image.md",
    
    # English docs - Integration
    "en-US/integration/dify.md",
    "en-US/integration/mcp-api.md",
    
    # English docs - Reference
    # "en-US/reference/how-to-configure-ollama.md",
    # "en-US/reference/HOW-TO-DEBUG.md",
    
    # Chinese docs - Design
    "zh-CN/design/architecture.md",
    "zh-CN/design/document_upload_design.md",
    "zh-CN/design/graph_index_creation.md",
    # "zh-CN/design/chat_history_design.md",
    
    # Chinese docs - Development
    "zh-CN/development/development-guide.md",
    
    # Chinese docs - Deployment
    "zh-CN/deployment/build-docker-image.md",
    
    # Chinese docs - Integration
    "zh-CN/integration/dify.md",
    "zh-CN/integration/mcp-api.md",
    
    # Chinese docs - Reference
    # "zh-CN/reference/how-to-configure-ollama.md",
    # "zh-CN/reference/HOW-TO-DEBUG.md",
]


def find_image_references(content: str) -> Set[str]:
    """
    Find all image references in markdown content.
    Matches patterns like: ![alt](path), <img src="path">, etc.
    """
    image_refs = set()
    
    # Match markdown image syntax: ![alt](path)
    md_pattern = r'!\[.*?\]\((.*?)\)'
    for match in re.finditer(md_pattern, content):
        image_refs.add(match.group(1))
    
    # Match HTML img tags: <img src="path">
    html_pattern = r'<img[^>]+src=["\']([^"\']+)["\']'
    for match in re.finditer(html_pattern, content):
        image_refs.add(match.group(1))
    
    return image_refs


def sync_file(src_path: Path, dest_path: Path) -> List[str]:
    """
    Sync a single file from docs/ to web/docs/
    Returns list of image paths referenced in the document.
    """
    # Create destination directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read source file
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find image references
    image_refs = find_image_references(content)
    
    # Copy file
    shutil.copy2(src_path, dest_path)
    print(f"✅ Synced: {src_path} -> {dest_path}")
    
    return list(image_refs)


def sync_images(doc_path: Path, image_refs: List[str]) -> None:
    """
    Sync images referenced in a document.
    """
    doc_dir = doc_path.parent
    
    for img_ref in image_refs:
        # Skip external URLs and absolute paths
        if img_ref.startswith(('http://', 'https://', '//', '/')):
            continue
        
        # Resolve image path relative to document
        img_src = (DOCS_ROOT / doc_dir / img_ref).resolve()
        img_dest = (WEB_DOCS_ROOT / doc_dir / img_ref).resolve()
        
        # Check if image exists
        if not img_src.exists():
            print(f"⚠️  Image not found: {img_src}")
            continue
        
        # Create destination directory
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        if not img_dest.exists() or os.path.getmtime(img_src) > os.path.getmtime(img_dest):
            shutil.copy2(img_src, img_dest)
            print(f"   📷 Image: {img_ref}")


def main():
    """
    Main sync function.
    """
    print("🔄 Syncing documentation from docs/ to web/docs/\n")
    
    # Check if directories exist
    if not DOCS_ROOT.exists():
        print(f"❌ Error: {DOCS_ROOT} directory not found")
        return 1
    
    # Create web/docs if it doesn't exist
    WEB_DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    
    synced_count = 0
    error_count = 0
    
    # Sync each file in whitelist
    for doc_path in SYNC_WHITELIST:
        src = DOCS_ROOT / doc_path
        dest = WEB_DOCS_ROOT / doc_path
        
        if not src.exists():
            print(f"⚠️  File not found in whitelist: {src}")
            error_count += 1
            continue
        
        try:
            # Sync document
            image_refs = sync_file(src, dest)
            synced_count += 1
            
            # Sync images
            if image_refs:
                sync_images(Path(doc_path), image_refs)
        
        except Exception as e:
            print(f"❌ Error syncing {doc_path}: {e}")
            error_count += 1
    
    # Sync _category.yaml files
    print("\n🔄 Syncing _category.yaml files...\n")
    category_count = 0
    for locale in ["en-US", "zh-CN"]:
        locale_dir = DOCS_ROOT / locale
        if locale_dir.exists():
            for category_dir in locale_dir.iterdir():
                if category_dir.is_dir():
                    category_file = category_dir / "_category.yaml"
                    if category_file.exists():
                        dest_file = WEB_DOCS_ROOT / locale / category_dir.name / "_category.yaml"
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(category_file, dest_file)
                        print(f"✅ Synced: {category_file} -> {dest_file}")
                        category_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"   ✅ Synced: {synced_count} documents")
    print(f"   ✅ Synced: {category_count} category files")
    if error_count > 0:
        print(f"   ❌ Errors: {error_count}")
    print(f"{'='*60}\n")
    
    print("💡 To sync more documents, edit SYNC_WHITELIST in scripts/sync-docs.py")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())
