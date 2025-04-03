#!/usr/bin/env bash

id=`date +"%Y-%m-%dT%T"`


jmeter -n -t tests/ApeRAG.jmx -l ./tests/results-${id} -e -o ./tests/report-${id}
