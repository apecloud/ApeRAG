{{/*
Expand the name of the chart.
*/}}
{{- define "aperag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "aperag.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "aperag.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aperag.labels" -}}
helm.sh/chart: {{ include "aperag.chart" . }}
{{ include "aperag.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "aperag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aperag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}


{{- define "api.labels" -}}
app.aperag.io/component: api
{{- end }}

{{- define "celeryworker.labels" -}}
app.aperag.io/component: celery-worker
{{- end }}

{{- define "frontend.labels" -}}
app.aperag.io/component: frontend
{{- end }}

{{- define "aperag.docray.serviceName" -}}
{{ include "aperag.fullname" . }}-docray
{{- end }}

{{/*
Construct DOCRAY_HOST value.
If docray.useExistingService is set, uses that value.
If docray.enabled is true, uses the docray service name.
Otherwise returns empty string.
*/}}
{{- define "aperag.docrayEndpoint" -}}
{{- if .Values.docray.useExistingService -}}
"{{ .Values.docray.useExistingService }}"
{{- else if .Values.docray.enabled -}}
"http://{{ include "aperag.docray.serviceName" . }}:8639"
{{- end }}
{{- end }}