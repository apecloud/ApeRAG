# Compose Runner

This runner is the first convenience launcher for the HTTP E2E suite.

Responsibilities:
- start the ApeRAG environment
- wait for `GET /health`
- leave execution of bootstrap and Hurl requests to other layers

Non-responsibilities:
- creating test users
- creating collections or documents
- embedding provider setup
- defining what smoke means

This runner is intentionally replaceable. The suite must remain runnable against future K8s-backed environments with the same bootstrap and Hurl files.
