#!/bin/bash

set -e

IMAGES_LIST_FILE="aperag-image-list.txt"

helm template --set paddler.enabled=false --set whisper.enabled=false deploy/aperag | grep 'image:' | awk '{print $NF}' | sort | uniq > $IMAGES_LIST_FILE

# 读取镜像列表文件，对每个镜像下载并保存到tar包中
while IFS= read -r image; do
    if [[ $image = \#* ||  -z "$image" ]]; then
        continue
    fi

    docker pull $image

done < "$IMAGES_LIST_FILE"

docker save $(cat $IMAGES_LIST_FILE) -o aperag-latest.tar.gz
