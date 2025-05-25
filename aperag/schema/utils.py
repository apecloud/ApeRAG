import json
from aperag.schema.view_models import CollectionConfig


def parseCollectionConfig(config: str) -> CollectionConfig:
    try:
        config_dict = json.loads(config)
        collection_config = CollectionConfig.parse_obj(config_dict)
        return collection_config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to parse collection config: {str(e)}")


def dumpCollectionConfig(collection_config: CollectionConfig) -> str:
    return collection_config.model_dump_json()