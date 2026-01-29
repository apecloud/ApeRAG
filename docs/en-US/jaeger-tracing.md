# Jaeger Distributed Tracing for ApeRAG

ApeRAG includes OpenTelemetry integration with Jaeger support for distributed tracing, allowing you to monitor and analyze request flows across your services.

## Quick Start

### 1. Enable Jaeger in Docker Compose

To start ApeRAG with Jaeger enabled:

```bash
# Start infrastructure with Jaeger
make compose-infra WITH_JAEGER=1

# Or start the full application stack with Jaeger
make compose-up WITH_JAEGER=1
```

Alternatively, you can use docker-compose directly:

```bash
# Start with Jaeger profile
docker-compose --profile jaeger up -d

# Or start specific services
docker-compose up -d jaeger postgres redis qdrant es
```

### 2. Enable Jaeger Tracing in Environment

Set the following environment variables:

```bash
JAEGER_ENABLED=True
JAEGER_ENDPOINT=http://aperag-jaeger:14268/api/traces  # Docker environment
# or
JAEGER_ENDPOINT=http://localhost:14268/api/traces      # Local development
```

### 3. Access Jaeger UI

Once Jaeger is running, access the web interface at:

- **Jaeger UI**: http://localhost:16686

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `True` | Enable/disable OpenTelemetry tracing |
| `OTEL_SERVICE_NAME` | `aperag` | Service name in traces |
| `OTEL_SERVICE_VERSION` | `1.0.0` | Service version in traces |
| `JAEGER_ENABLED` | `False` | Enable/disable Jaeger exporter |
| `JAEGER_ENDPOINT` | - | Jaeger collector endpoint |
| `OTEL_CONSOLE_ENABLED` | `False` | Enable console span output |
| `OTEL_FASTAPI_ENABLED` | `True` | Enable FastAPI instrumentation |
| `OTEL_SQLALCHEMY_ENABLED` | `True` | Enable SQLAlchemy instrumentation |
| `OTEL_MCP_ENABLED` | `True` | Enable MCP agent trace injection |

### Jaeger Ports

| Port | Service | Description |
|------|---------|-------------|
| 16686 | Web UI | Jaeger query and visualization interface |
| 14268 | HTTP | HTTP collector for receiving spans |
| 14250 | gRPC | gRPC collector for receiving spans |
| 6831 | UDP | UDP agent port (legacy) |
| 6832 | UDP | UDP agent port (legacy) |

## What Gets Traced

ApeRAG automatically instruments:

1. **HTTP Requests** (FastAPI)
   - All API endpoints
   - Request/response details
   - Error tracking

2. **Database Operations** (SQLAlchemy)
   - SQL queries
   - Database connection info
   - Query performance

3. **MCP Agent Events**
   - Agent interactions
   - Tool usage
   - Session tracking

4. **Custom Application Spans**
   - LLM API calls
   - Document processing
   - Graph operations

## Using Traces

### 1. Find Traces by Service

In Jaeger UI:
- Select "aperag" from the Service dropdown
- Choose an operation (e.g., "GET /api/v1/collections")
- Click "Find Traces"

### 2. Analyze Request Flow

Each trace shows:
- **Timeline**: Visual representation of span duration
- **Service Map**: How services interact
- **Error Detection**: Failed operations highlighted in red
- **Performance**: Identify slow operations

### 3. Debugging

For debugging specific issues:
- Search traces by operation name
- Filter by tags (user_id, collection_id, etc.)
- Look for error spans (marked in red)
- Examine span logs for detailed error information

## Development Usage

### Local Development

For local development without Docker:

```bash
# Start Jaeger locally
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 14268:14268 \
  jaegertracing/all-in-one:1.60

# Set environment variables
export JAEGER_ENABLED=True
export JAEGER_ENDPOINT=http://localhost:14268/api/traces

# Start your application
make run-backend
```

### Custom Tracing

Add custom spans to your code:

```python
from aperag.trace import get_tracer, create_span

tracer = get_tracer(__name__)

# Synchronous function
with create_span(tracer, "my_operation", custom_attr="value"):
    # Your code here
    pass

# Or use decorators
from aperag.trace import trace_async_function

@trace_async_function("custom_operation")
async def my_async_function():
    # This function will be automatically traced
    pass
```

## Production Considerations

### Performance Impact

- Tracing adds minimal overhead (~1-5ms per request)
- Sampling can be configured to reduce load
- Consider disabling console output in production

### Data Retention

- Jaeger stores traces in memory by default
- For production, consider using persistent storage:
  - Elasticsearch backend
  - Cassandra backend
  - Kafka for high-throughput scenarios

### Security

- Jaeger UI has no built-in authentication
- Consider putting it behind a reverse proxy with auth
- Traces may contain sensitive data - review what gets traced

## Troubleshooting

### Jaeger Not Receiving Traces

1. Check if Jaeger is running: `docker ps | grep jaeger`
2. Verify endpoint configuration: `echo $JAEGER_ENDPOINT`
3. Check application logs for tracing errors
4. Ensure `OTEL_ENABLED=True` and `JAEGER_ENABLED=True`

### Missing Spans

1. Verify instrumentation is enabled for the component
2. Check if the operation is being captured
3. Look for exceptions in span creation

### Performance Issues

1. Disable console output: `OTEL_CONSOLE_ENABLED=False`
2. Configure sampling rates if needed
3. Monitor Jaeger resource usage

## Resources

- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [FastAPI OpenTelemetry](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html) 