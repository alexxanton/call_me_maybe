from pydantic import BaseModel, ConfigDict, StringConstraints
from typing import Dict, Literal, Annotated


Name = Annotated[
    str, StringConstraints(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
]


class Parameter(BaseModel):
    """Represents a function parameter or return value."""
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "number",
        "num",
        "string",
        "boolean",
        "bool",
        "integer",
        "int",
        "float"
    ]


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
