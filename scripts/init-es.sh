#!/bin/bash

# Function to check if ES is healthy
is_healthy() {
    curl -s -X GET "http://localhost:9200/_cluster/health" | grep -q "green\|yellow"
}

# Function to check if IK Analyzer is installed
ik_plugin_installed() {
    [ -d "/usr/share/elasticsearch/plugins/analysis-ik" ]
}

# Function to check if ES is ready
check_ready() {
    is_healthy
}

if [ "$1" = "check" ]; then
    check_ready
    exit $?
fi

# Check and install IK Analyzer if needed
if ! ik_plugin_installed; then
    echo "Installing IK Analyzer..."
    /usr/share/elasticsearch/bin/elasticsearch-plugin install -b https://get.infini.cloud/elasticsearch/analysis-ik/8.8.2
    if [ "$?" -ne 0 ]; then
        echo "Failed to install IK Analyzer"
        exit 1
    fi
    echo "IK Analyzer installed successfully"
else
    echo "IK Analyzer is already installed"
fi

# Start ES in foreground
echo "Starting Elasticsearch..."
exec /usr/share/elasticsearch/bin/elasticsearch
