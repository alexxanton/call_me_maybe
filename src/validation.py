from pydantic import BaseModel, ConfigDict
from typing import Dict, Literal


class Parameter(BaseModel):
    """Represents a function parameter or return value."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["number", "string"]


class Function(BaseModel):
    """Defines a callable function and its parameters."""
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter


class Prompt(BaseModel):
    """Represents a prompt with instructions for the model."""
    model_config = ConfigDict(extra="forbid")
    prompt: str
