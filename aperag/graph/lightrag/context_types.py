from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


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
