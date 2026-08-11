from pydantic import BaseModel, ConfigDict, StringConstraints
from typing import Dict, Literal, Annotated


Name = Annotated[
    str, StringConstraints(pattern=r"^[a-zA-z_][a-zA-z0-9_]*$")
]


class Parameter(BaseModel):
    """Represents a function parameter or return value."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["number", "string"]


class Function(BaseModel):
    """Defines a callable function and its parameters."""
    model_config = ConfigDict(extra="forbid")
    name: Name
    description: str
    parameters: Dict[Name, Parameter]
    returns: Parameter


class Prompt(BaseModel):
    """Represents a prompt with instructions for the model."""
    model_config = ConfigDict(extra="forbid")
    prompt: str
