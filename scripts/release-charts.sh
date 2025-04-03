#!/usr/bin/env bash

# set variable from environment variables and set default value
VERSION=${VERSION:-0.1.0}

CHART_REPO=https://jihulab.com/api/v4/projects/85949/packages/helm/api/stable/charts

CHART_REPO_USER=${CHART_REPO_USER}
if [ -z "$CHART_REPO_USER" ]; then
  echo "CHART_REPO_USER is not set"
  exit 1
fi

CHART_REPO_PASSWD=${CHART_REPO_PASSWD}
if [ -z "$CHART_REPO_PASSWD" ]; then
  echo "CHART_REPO_PASSWD is not set"
  exit 1
fi

function upload() {
  local chart=$1

  echo "Uploading $chart-${VERSION} to ${CHART_REPO}"

  curl --request POST --form "chart=@${chart}-${VERSION}.tgz" --user "${CHART_REPO_USER}":"${CHART_REPO_PASSWD}" ${CHART_REPO}

  echo ""
}

helm package --version "${VERSION}" ./deploy/aperag
upload aperag

helm package --version "${VERSION}" ./deploy/llmserver
upload llmserver