# task #17 sub-task #21 (probe layering) acceptance contract pin —
# the API-side hurl smoke must verify the three probe endpoints emerge
# after the API/Worker hard cut deployment split:
#
#   /health             — kept for backward-compat with existing
#                         k8s probes / external monitors.
#   /health/live        — liveness; only proves the process is alive,
#                         must not touch PG / Redis / Qdrant.
#   /health/ready       — readiness; proves the HTTP entry can accept
#                         requests, must not consume the main DB pool.
#   /health/diagnostics — deep-dependency check; gated by admin auth /
#                         intranet, NOT asserted in smoke (Planetegg
#                         covers it from the release-validation lane
#                         per msg=64f33ceb).

GET {{base_url}}/health
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.status" == "healthy"
jsonpath "$.service" == "aperag-api"

GET {{base_url}}/health/live
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.status" == "healthy"

GET {{base_url}}/health/ready
HTTP 200
[Asserts]
header "content-type" contains "application/json"
jsonpath "$.status" == "ready"
