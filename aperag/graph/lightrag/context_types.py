from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from aperag.graph.lightrag.prompt import GRAPH_FIELD_SEP


class LightRagEntityContext(BaseModel):
    id: str = Field(..., description="Unique identifier for the entity.")
    entity: str = Field(..., description="The name or text content of the entity.")
    type: str = Field(..., description="The type of the entity (e.g., 'event', 'person', 'location').")
    description: Optional[str] = Field(None, description="A detailed description of the entity.")
    rank: Optional[int] = Field(None, description="The rank or importance level of the entity.")
    created_at: datetime = Field(
        ...,
        description=(
            "Timestamp when the entity context object was created. "
            "Must be provided from the JSON source (e.g., 'YYYY-MM-DD HH:MM:SS' format)."
        ),
    )
    file_path: Optional[List[str]] = Field(
        None, description="A list of file paths where the entity information originated."
    )


class LightRagRelationContext(BaseModel):
    id: str = Field(..., description="Unique identifier for the relation.")
    entity1: str = Field(..., description="The name or ID of the first entity involved in the relation.")
    entity2: str = Field(..., description="The name or ID of the second entity involved in the relation.")
    description: Optional[str] = Field(None, description="A detailed description of the relation.")
    keywords: Optional[str] = Field(
        None, description="Keywords associated with the relation, typically comma-separated."
    )
    weight: Optional[float] = Field(
        None, description="A numerical weight indicating the strength or importance of the relation."
    )
    rank: Optional[int] = Field(None, description="The rank or importance level of the relation.")
    created_at: datetime = Field(
        ...,
        description=(
            "Timestamp when the relation context object was created. "
            "Must be provided from the JSON source (e.g., 'YYYY-MM-DD HH:MM:SS' format)."
        ),
    )
    file_path: Optional[List[str]] = Field(
        None, description="A list of file paths where the entity information originated."
    )


class LightRagTextUnitContext(BaseModel):
    id: str = Field(..., description="Unique identifier for this text chunk.")
    content: str = Field(..., description="The raw textual content.")
    file_path: Optional[List[str]] = Field(
        None, description="A list of file paths where the entity information originated."
    )


# Conversion functions from LightRAG JSON format to Pydantic models


def json_to_entity_context(json_data: dict) -> LightRagEntityContext:
    """
    Convert LightRAG entity JSON to LightRagEntityContext.

    Args:
        json_data: Dict with keys like 'id', 'entity', 'type', 'description',
                  'rank', 'created_at', 'file_path'

    Returns:
        LightRagEntityContext instance
    """
    # Parse datetime from string
    created_at = datetime.strptime(json_data["created_at"], "%Y-%m-%d %H:%M:%S")

    # Parse file_path - split by <SEP> if it's a string, or keep as list if already a list
    file_path = None
    if json_data.get("file_path"):
        if isinstance(json_data["file_path"], str):
            file_path = json_data["file_path"].split(GRAPH_FIELD_SEP)
        elif isinstance(json_data["file_path"], list):
            file_path = json_data["file_path"]

    return LightRagEntityContext(
        id=json_data["id"],
        entity=json_data["entity"],
        type=json_data["type"],
        description=json_data.get("description"),
        rank=json_data.get("rank"),
        created_at=created_at,
        file_path=file_path,
    )


def json_to_relation_context(json_data: dict) -> LightRagRelationContext:
    """
    Convert LightRAG relation JSON to LightRagRelationContext.

    Args:
        json_data: Dict with keys like 'id', 'entity1', 'entity2', 'description',
                  'keywords', 'weight', 'rank', 'created_at', 'file_path'

    Returns:
        LightRagRelationContext instance
    """
    # Parse datetime from string
    created_at = datetime.strptime(json_data["created_at"], "%Y-%m-%d %H:%M:%S")

    # Parse file_path - split by <SEP> if it's a string, or keep as list if already a list
    file_path = None
    if json_data.get("file_path"):
        if isinstance(json_data["file_path"], str):
            file_path = json_data["file_path"].split(GRAPH_FIELD_SEP)
        elif isinstance(json_data["file_path"], list):
            file_path = json_data["file_path"]

    return LightRagRelationContext(
        id=json_data["id"],
        entity1=json_data["entity1"],
        entity2=json_data["entity2"],
        description=json_data.get("description"),
        keywords=json_data.get("keywords"),
        weight=json_data.get("weight"),
        rank=json_data.get("rank"),
        created_at=created_at,
        file_path=file_path,
    )


def json_to_text_unit_context(json_data: dict) -> LightRagTextUnitContext:
    """
    Convert LightRAG text unit JSON to LightRagTextUnitContext.

    Args:
        json_data: Dict with keys like 'id', 'content', 'file_path'

    Returns:
        LightRagTextUnitContext instance
    """
    # Parse file_path - split by <SEP> if it's a string, or keep as list if already a list
    file_path = None
    if json_data.get("file_path"):
        if isinstance(json_data["file_path"], str):
            file_path = json_data["file_path"].split(GRAPH_FIELD_SEP)
        elif isinstance(json_data["file_path"], list):
            file_path = json_data["file_path"]

    return LightRagTextUnitContext(id=json_data["id"], content=json_data["content"], file_path=file_path)


def json_list_to_entity_contexts(json_list: List[dict]) -> List[LightRagEntityContext]:
    return [json_to_entity_context(json_data) for json_data in json_list]


def json_list_to_relation_contexts(json_list: List[dict]) -> List[LightRagRelationContext]:
    return [json_to_relation_context(json_data) for json_data in json_list]


def json_list_to_text_unit_contexts(json_list: List[dict]) -> List[LightRagTextUnitContext]:
    return [json_to_text_unit_context(json_data) for json_data in json_list]
