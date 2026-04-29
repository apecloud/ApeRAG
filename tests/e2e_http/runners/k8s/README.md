# K8s Runner

This runner keeps the same bootstrap and Hurl suite while swapping only the launcher.

Current mode:
- default behavior uses `kubectl port-forward` against a target service
- the runner then waits for `GET /health/ready`
- `down.sh` tears down the background port-forward process

Key environment variables:
- `E2E_K8S_NAMESPACE`
- `E2E_K8S_SERVICE`
- `E2E_K8S_REMOTE_PORT`
- `E2E_K8S_LOCAL_PORT`
- `E2E_K8S_USE_PORT_FORWARD=0` if an ingress URL is already exposed and `E2E_BASE_URL` is set

The suite contract remains unchanged:
- bootstrap protocol
- Hurl files
- testdata layout
