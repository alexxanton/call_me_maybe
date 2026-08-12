from pydantic import BaseModel
from .validation import Function, Parameter
from typing import List, Iterator, Dict, Set, Tuple


class ConstrainedDecoder(BaseModel):
    """Contains the logic for constrained decoding."""

    prompt: str
    functions: List[Function]

    def __init__(self, prompt: str, functions: List[Function]) -> None:
        """Initialize the constrained decoder."""
        super().__init__(prompt=prompt, functions=functions)

        self._prefix = '"name": "'
        self._func_names = [f.name for f in functions]
        self._states = iter([
            f'{{\n  "prompt": "{prompt}",\n  "name": "',
            ',\n  "parameters": {\n',
            '  }\n}'
        ])
        self._state = next(self._states)
        self._finished = False
        self._func_name = ""
        self._last_param_reached = False

    def _get_params(self, name: str) -> Iterator[Tuple[str, Parameter]]:
        """Get parameters from function."""
        func = next((f for f in self.functions if f.name == name), None)
        if func is None:
            return iter([])
        return iter(func.parameters.items())

    def retrieve_func_name(self, output: str) -> None:
        """Get function name from generated output."""
        name = output.split(self._prefix)[-1][:-1]
        self._state = next(self._states)
        self._params = self._get_params(name)
        self._func_name = name

    def inject_next_param(self) -> str:
        """Inject the next parameter from the function."""
        param = next(self._params, None)
        if param is None:
            self._finished = True
            return ""
        formatted_param = (
            f'    "{param[0]}": ' +
            ('"' if param[1].type == "string" else "")
        )
        return formatted_param

    def get_name_tokens(
            self, output: str, vocab: Dict[str, int], id_values: Dict[int, str]
    ) -> Set[int]:
        """Get allowed tokens for function name."""
        name = output.split(self._prefix)[-1]
        allowed = set()
        for key, val in id_values.items():
            candidate = name + val
            if candidate in self._func_names:
                allowed.add(key)
            elif any(n.startswith(candidate) for n in self._func_names):
                allowed.add(key)

            if name in self._func_names:
                dquote = vocab.get('"')
                if dquote is not None:
                    allowed.add(dquote)
        return allowed

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
