from aperag.schema.common import CollectionConfig
from aperag.platform.source.base import CustomSourceInitializationError, get_source
from aperag.platform.source.upload import UploadSource
from aperag.views.utils import validate_source_connect_config


def test_collection_config_defaults_to_system_source():
    config = CollectionConfig()

    assert config.source == "system"


def test_get_source_only_allows_system_source():
    source = get_source(CollectionConfig(source="system"))

    assert isinstance(source, UploadSource)


def test_get_source_rejects_removed_legacy_sources():
    try:
        get_source(CollectionConfig(source="git"))
    except CustomSourceInitializationError as exc:
        assert str(exc) == "unsupported collection source: git"
    else:
        raise AssertionError("expected CustomSourceInitializationError")


def test_validate_source_connect_config_rejects_removed_legacy_sources():
    is_valid, error = validate_source_connect_config(CollectionConfig(source="git"))

    assert is_valid is False
    assert error == "unsupported collection source: git"
