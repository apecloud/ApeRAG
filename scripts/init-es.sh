#!/bin/bash

# Function to check if ES is ready
wait_for_es() {
    until curl -s -X GET "http://localhost:9200/_cluster/health" | grep -q "green\|yellow"; do
        echo "Waiting for Elasticsearch to be ready..."
        sleep 5
    done
    echo "Elasticsearch is ready!"
}

# Start ES in background
/usr/share/elasticsearch/bin/elasticsearch &
ES_PID=$!

# Wait for ES to be ready
wait_for_es

# Check and install IK Analyzer if needed
if [ ! -d "/usr/share/elasticsearch/plugins/ik" ]; then
    echo "Installing IK Analyzer..."
    /usr/share/elasticsearch/bin/elasticsearch-plugin install -b https://get.infini.cloud/elasticsearch/analysis-ik/8.8.2
    if [ "$?" -ne 0 ]; then
        echo "Failed to install IK Analyzer"
        exit 1
    fi
    echo "IK Analyzer installed successfully"

    # Kill the current ES process
    kill $ES_PID
    wait $ES_PID

    # Start ES again
    echo "Restarting Elasticsearch after plugin installation..."
    exec /usr/share/elasticsearch/bin/elasticsearch
else
    echo "IK Analyzer is already installed"
    # Keep the current ES process running
    wait $ES_PID
fi