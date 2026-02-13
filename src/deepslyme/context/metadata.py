from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Type, Union

# Re-export upstream metadata keys
from slyme.context.metadata import HELP, TYPE

__all__ = [
    "ARG",
    "Arg",
    "HELP",
    "TYPE",
]

# Metadata Key
ARG = "node.arg"

_Missing = Enum("_Missing", ["MARK"])
_MISSING = _Missing.MARK


@dataclass(frozen=True)
class Arg:
    """
    Argument definition to be used in Ref metadata.
    Designed to be minimal and universal, compatible with argparse, hydra, etc.

    Example:
        Ref("model.lr", metadata={ARG: Arg(default=1e-3, help="Learning rate")})
    """

    default: Any = _MISSING
    default_factory: Union[Callable[[], Any], _Missing] = _MISSING
    help: Optional[str] = None
    type: Optional[Type] = None
    choices: Optional[Iterable[Any]] = None
    required: bool = False
    nargs: Union[str, int, None] = None
    aliases: List[str] = field(default_factory=list)
    metavar: Optional[str] = None

    @staticmethod
    def is_missing(value: Any) -> bool:
        return value is _MISSING

    def resolve_default(self) -> Any:
        if self.default is not _MISSING:
            return self.default
        if self.default_factory is not _MISSING:
            return self.default_factory()
        return _MISSING
