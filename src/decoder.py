import re
import numpy as np
from pydantic import BaseModel
from .validation import Function, Parameter
from typing import List, Iterator, Dict, Set, Tuple, Optional, ClassVar


class ConstrainedDecoder(BaseModel):
    """Contains the logic for constrained decoding."""

    prompt: str
    functions: List[Function]
    vocab: Dict[str, int]
    _func_tokens: ClassVar[Optional[Dict[int, str]]] = None
    _ids: ClassVar[Optional[Dict[int, str]]] = None
    _numbers : ClassVar[Set[str]] = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    }

    def __init__(
        self, prompt: str, functions: List[Function], vocab: Dict[str, int]
    ) -> None:
        """Initialize the constrained decoder."""
        prompt = prompt.replace("\\", "\\\\").replace('"', '\\"')
        super().__init__(prompt=prompt, functions=functions, vocab=vocab)

        self._prefix = '"name": "'
        self._func_names = [f.name for f in functions]
        self._states = iter([
            f'{{\n  "prompt": "{prompt}",\n  "name": "',
            ',\n  "parameters": {\n',
            '  }\n}'
        ])
        self._state = next(self._states)
        self._next_param: Optional[Tuple[str, Parameter]] = None
        self._finished = False
        self._func_name = ""
        self._last_param_reached = False
        self._params: Iterator[Tuple[str, Parameter]] = iter(())
        self._name_complete = False
        self._param_complete = True
        self._param_type = ""

        if ConstrainedDecoder._ids is None:
            ConstrainedDecoder._ids = {v: k for k, v in vocab.items()}

        if ConstrainedDecoder._func_tokens is None:
            ConstrainedDecoder._func_tokens = {
                v: k for k, v in vocab.items()
                if re.fullmatch("[A-Za-z0-9_]+", k)
                and any(k in f for f in self._func_names)
            }

        self._alpha_num = ConstrainedDecoder._func_tokens
        self._id_values = ConstrainedDecoder._ids

    def _get_params(self, name: str) -> None:
        """Get parameters from function."""
        func = next((f for f in self.functions if f.name == name), None)
        if func is None:
            return
        self._params = iter(func.parameters.items())
        self._next_param = next(self._params, None)

    def retrieve_func_name(self, output: str) -> None:
        """Get function name from generated output."""
        name = output.split(self._prefix)[-1][:-1]
        self._state = next(self._states)
        self._get_params(name)
        self._func_name = name

    def _get_type_tokens(self) -> Set[str]:
        """Get allowed tokens for each type."""
        match self._param_type:
            case "integer" | "int":
                return ConstrainedDecoder._numbers
            case "number" | "float" | "num":
                return ConstrainedDecoder._numbers | {"-", "."}
            case "boolean" | "bool":
                return {"true", "false"}
            case _:
                return set()
        return set()

    def get_allowed_tokens(
        self, logits: np.array = None, allowed = set(), **kwargs
    ) -> Set[int]:
        """Get allowed tokens for current parameter."""
        if "reset" in kwargs and kwargs["reset"]:
            allowed.clear()
            return set()

        if "output" in kwargs:
            allowed_tokens = self.get_name_tokens(kwargs["output"])

        newline = "Ċ" if self._last_param_reached else ",Ċ"

        if not allowed and self._name_complete:
            selected_tokens = self._get_type_tokens()
            if not selected_tokens:
                return logits
            allowed.update(selected_tokens)

        valid_logits = np.full_like(logits, -float("inf"))

        if self._name_complete:
            allowed_tokens = {self.vocab[n] for n in allowed | {newline}}

        for allowed in allowed_tokens:
            valid_logits[allowed] = logits[allowed]
        return valid_logits

    def get_name_tokens(self, output: str) -> Set[int]:
        """Get allowed tokens for function name."""
        name = output.split(self._prefix)[-1]
        allowed = set()
        for key, val in self._alpha_num.items():
            candidate = name + val
            if candidate in self._func_names:
                allowed.add(key)
            elif any(n.startswith(candidate) for n in self._func_names):
                allowed.add(key)

            if name in self._func_names:
                dquote = self.vocab.get('"')
                if dquote is not None:
                    allowed.add(dquote)
        return allowed

    def inject_next_param(self) -> str:
        """Inject the next parameter from the function."""
        self._param_type = ""
        self.get_allowed_tokens(reset=True) #  Rewrite allowed tokens set.
        param = self._next_param
        if param is None:
            return ""
        self._next_param = next(self._params, None)
        self._param_type = param[1].type
        if self._next_param is None:
            self._last_param_reached = True
        formatted_param = (
            f'    "{param[0]}": ' +
            ('"' if param[1].type == "string" else "")
        )
        return formatted_param

    def close_param(self) -> None:
        """Finalize parameter generation."""
        if self._last_param_reached:
            self._finished = True
            self._state = next(self._states)

    @property
    def state(self) -> str:
        """Get decoder state."""
        return self._state

    @property
    def func_name(self) -> str:
        """Get selected function name."""
        return self._func_name

    @property
    def finished(self) -> bool:
        """Get decoder finish status."""
        return self._finished

    @property
    def name_complete(self) -> bool:
        """Get name completion state."""
        return self._name_complete

    @name_complete.setter
    def name_complete(self, b: bool) -> None:
        """Set name completion state."""
        self._name_complete = b

    @property
    def param_complete(self) -> bool:
        """Get param completion state."""
        return self._param_complete

    @param_complete.setter
    def param_complete(self, b: bool) -> None:
        """Set param completion state."""
        self._param_complete = b
