from pydantic import BaseModel
from typing import Dict, Literal


class Parameter(BaseModel):
    """Represents a function parameter or return value."""
    type: Literal["number", "string"]


class Function(BaseModel):
    """Defines a callable function and its parameters."""
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter


class Prompt(BaseModel):
    """Represents a prompt with instructions for the model."""
    prompt: str
