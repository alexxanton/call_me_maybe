import re
from pydantic import BaseModel
from .validation import Function, Parameter
from typing import List, Iterator, Dict, Set, Tuple, Optional, ClassVar


class ConstrainedDecoder(BaseModel):
    """Contains the logic for constrained decoding."""

    prompt: str
    functions: List[Function]
    vocab: Dict[str, int]

    def __init__(
        self, prompt: str, functions: List[Function], vocab: Dict[str, int]
    ) -> None:
        """Initialize the constrained decoder."""
        prompt = prompt.replace('"', '\\"')
        super().__init__(prompt=prompt, functions=functions, vocab=vocab)

        self._id_values = {v: k for k, v in vocab.items()}
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
        self._params: Optional[Iterator[Tuple[str, Parameter]]] = None
        self._name_complete = False
        self._param_complete = True

        self._alpha_num = {
            v: k for k, v in vocab.items()
            if re.fullmatch("[A-Za-z0-9_]+", k)
            and any(k in f for f in self._func_names)
        }
        print(len(self._alpha_num))

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

    def get_num_tokens(
        self, vocab: Dict[str, int], id_values: Dict[int, str]
    ) -> Set[int]:
        """Get alowed tokens for number parameters."""
        allowed = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", "-", "."
        }

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
        param = self._next_param
        if param is None:
            return ""
        self._next_param = next(self._params, None)
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
