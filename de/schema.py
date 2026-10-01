from pydantic import BaseModel
from typing import Literal, Optional

class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    source_type: Literal["rules", "faq"]
    priority: Literal["official", "community"]

    section_id: Optional[str] = None       
    section_path: Optional[str] = None      # "Часть 1 > 105. Базовые свойства > 105.2. Элитность"


    card_name: Optional[str] = None
    card_number: Optional[int] = None
    set_name: Optional[str] = None          # "4. Ложные боги"

    language: str = "ru"
    updated_at: str

class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata