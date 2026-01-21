from pydantic import BaseModel

class ExtractRequest(BaseModel):
    request_id: str