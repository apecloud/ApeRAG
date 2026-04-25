# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

ObservabilityMode = Literal["local", "off", "otlp", "collector"]


@dataclass(frozen=True)
class ObservabilityConfig:
    mode: ObservabilityMode = "local"
    service_name: str = "aperag"
    service_version: str = "1.0.0"
    environment: str = "development"
    log_format: Literal["json", "text"] = "json"
    capture_content: bool = False
    sample_ratio: float = 1.0
    otlp_endpoint: Optional[str] = None
    otlp_headers: dict[str, str] = field(default_factory=dict)
    otlp_protocol: str = "http/protobuf"
    console_enabled: bool = False
    enable_fastapi: bool = True
    enable_sqlalchemy: bool = True
    enable_mcp: bool = True
    resource_attributes: dict[str, str] = field(default_factory=dict)

    @property
    def tracing_enabled(self) -> bool:
        return self.mode != "off"

    @property
    def remote_export_enabled(self) -> bool:
        return self.mode in {"otlp", "collector"} and bool(self.otlp_endpoint)

    @property
    def metrics_enabled(self) -> bool:
        return self.mode in {"otlp", "collector"} and bool(self.otlp_endpoint)

    @property
    def uses_otlp(self) -> bool:
        return self.remote_export_enabled

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        from aperag.config import settings

        return build_observability_config(settings)


def normalize_mode(value: str | None) -> ObservabilityMode:
    mode = (value or "local").strip().lower()
    if mode in {"false", "0", "disabled", "disable", "none"}:
        return "off"
    if mode in {"true", "1", "enabled", "enable"}:
        return "local"
    if mode in {"local", "off", "otlp", "collector"}:
        return mode  # type: ignore[return-value]
    return "local"


def clamp_sample_ratio(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def parse_otlp_headers(value: str | Mapping[str, str] | None) -> dict[str, str]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        return {str(key).strip(): str(val).strip() for key, val in value.items() if str(key).strip()}

    headers: dict[str, str] = {}
    for item in str(value).split(","):
        if not item.strip() or "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        if key:
            headers[key] = val.strip()
    return headers


def parse_resource_attributes(value: str | None) -> dict[str, str]:
    return parse_otlp_headers(value)


def parse_legacy_bool(value) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def build_observability_config(settings=None) -> ObservabilityConfig:
    if settings is None:
        from aperag.config import settings as app_settings

        settings = app_settings

    explicit_mode = getattr(settings, "aperag_observability_mode", None)
    mode = normalize_mode(explicit_mode)
    legacy_otel_enabled = parse_legacy_bool(getattr(settings, "otel_enabled", None))
    if explicit_mode in (None, "") and legacy_otel_enabled is False:
        mode = "off"

    return ObservabilityConfig(
        mode=mode,
        service_name=getattr(settings, "otel_service_name", "aperag"),
        service_version=getattr(settings, "otel_service_version", "1.0.0"),
        environment=getattr(settings, "otel_environment", "development"),
        log_format=(getattr(settings, "aperag_observability_log_format", "json") or "json").lower(),
        capture_content=bool(getattr(settings, "aperag_observability_capture_content", False)),
        sample_ratio=clamp_sample_ratio(float(getattr(settings, "aperag_observability_sample_ratio", 1.0))),
        otlp_endpoint=getattr(settings, "otel_exporter_otlp_endpoint", None),
        otlp_headers=parse_otlp_headers(getattr(settings, "otel_exporter_otlp_headers", None)),
        otlp_protocol=getattr(settings, "otel_exporter_otlp_protocol", "http/protobuf") or "http/protobuf",
        console_enabled=bool(getattr(settings, "otel_console_enabled", False)),
        enable_fastapi=bool(getattr(settings, "otel_fastapi_enabled", True)),
        enable_sqlalchemy=bool(getattr(settings, "otel_sqlalchemy_enabled", True)),
        enable_mcp=bool(getattr(settings, "otel_mcp_enabled", True)),
        resource_attributes=parse_resource_attributes(getattr(settings, "otel_resource_attributes", None)),
    )
