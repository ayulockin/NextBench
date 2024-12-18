from typing import Optional
from pydantic import BaseModel


class RequestResult(BaseModel):
    success: bool  # If the request was successful (True)
    cached: bool  # If the result was cached (True)
    error: Optional[str] = None # If the request was not successful, the error message
    completions: Optional[list[str]]  # If the request was successful, the completions
