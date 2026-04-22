#!/bin/bash

# Function to check if ES is healthy
is_healthy() {
    curl -s -X GET "http://localhost:9200/_cluster/health" | grep -q "green\|yellow"
}

# Function to check if IK Analyzer is installed
ik_plugin_installed() {
    [ -d "/usr/share/elasticsearch/plugins/analysis-ik" ]
}

require_ik_plugin() {
    [ "${ES_REQUIRE_IK_PLUGIN:-true}" = "true" ]
}

auto_install_ik_plugin() {
    [ "${ES_AUTO_INSTALL_IK:-false}" = "true" ]
}

# Function to check if ES is ready
check_ready() {
    is_healthy || return 1

    if require_ik_plugin && ! ik_plugin_installed; then
        echo "IK Analyzer is required but not installed"
        return 1
    fi

    return 0
}

if [ "$1" = "check" ]; then
    check_ready
    exit $?
fi

# Check and install IK Analyzer if needed
if require_ik_plugin && ! ik_plugin_installed; then
    if auto_install_ik_plugin; then
        IK_PLUGIN_URL="${ES_IK_PLUGIN_URL:-https://get.infini.cloud/elasticsearch/analysis-ik/8.8.2}"
        echo "Installing IK Analyzer from ${IK_PLUGIN_URL}..."
        /usr/share/elasticsearch/bin/elasticsearch-plugin install -b "${IK_PLUGIN_URL}"
        if [ "$?" -ne 0 ]; then
            echo "Failed to install IK Analyzer"
            exit 1
        fi
        echo "IK Analyzer installed successfully"
    else
        echo "IK Analyzer is required but not installed."
        echo "Use an Elasticsearch image with analysis-ik pre-baked, or explicitly set ES_AUTO_INSTALL_IK=true."
        exit 1
    fi
elif require_ik_plugin; then
    echo "IK Analyzer is already installed"
else
    echo "IK Analyzer requirement disabled"
fi

# Start ES in foreground
echo "Starting Elasticsearch..."
exec /usr/share/elasticsearch/bin/elasticsearch
