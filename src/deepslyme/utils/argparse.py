"""
Argument parsing utilities for slyme, designed to work with Ref metadata.
"""

import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Union,
    Type,
    get_origin,
    get_args,
    Literal,
    Dict,
    Tuple,
)
from slyme.utils.store import Ref, MISSING, Missing

__all__ = [
    "ARG",
    "Arg",
    "populate_parser",
    "parse_refs",
]

# Metadata Key
ARG = "node.arg"


def _string_to_bool(v: Union[str, bool]) -> bool:
    """
    Helper to convert string to boolean for argparse.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif v_lower in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Expected boolean value, got {v!r}. (Valid: yes/no, true/false, t/f, y/n, 1/0)"
        )


@dataclass
class Arg:
    """
    Argument definition to be used in Ref metadata.
    
    Example:
        Ref("model.lr", metadata={ARG: Arg(default=1e-3, help="Learning rate")})
    """
    default: Any = MISSING
    default_factory: Union[Callable[[], Any], Missing] = MISSING
    help: Optional[str] = None
    type: Optional[Type] = None
    choices: Optional[Iterable[Any]] = None
    required: bool = False
    nargs: Union[str, int, None] = None
    aliases: List[str] = field(default_factory=list)
    metavar: Optional[str] = None

    def resolve_default(self) -> Any:
        if self.default is not MISSING:
            return self.default
        if self.default_factory is not MISSING:
            return self.default_factory()
        return MISSING


def _infer_type(arg: Arg, ref: Ref) -> Type:
    """
    Infer the type of the argument based on explicit definition or default value.
    """
    if arg.type is not None:
        return arg.type
    
    default_val = arg.resolve_default()
    if default_val is not MISSING and default_val is not None:
        return type(default_val)
    
    # Fallback: if no type info is available, assume str
    return str


def _add_argument_for_ref(parser: argparse.ArgumentParser, ref: Ref, arg: Arg):
    """
    Add a single Ref argument to the parser.
    """
    # 1. Determine Flag Names
    # Convert dotted path "model.config.lr" -> "--model.config.lr" and "--model-config-lr"
    path = ref.path
    flags = [f"--{path}"]
    if "." in path:
        flags.append(f"--{path.replace('.', '-')}")
    if "_" in path:
         flags.append(f"--{path.replace('_', '-')}")
    
    # Add aliases (e.g., "-lr")
    flags.extend(arg.aliases)
    
    # Remove duplicates while preserving order
    flags = list(dict.fromkeys(flags))

    # 2. Base kwargs for argparse
    kwargs = {
        "dest": path, # Keep the dot notation for the destination key
        "help": arg.help,
        "metavar": arg.metavar,
    }

    # 3. Resolve Type and Default
    arg_type = _infer_type(arg, ref)
    default_val = arg.resolve_default()

    # Handle Optional[T] or Union[T, None] (Primitive unpacking)
    origin = get_origin(arg_type)
    args = get_args(arg_type)
    if origin is Union and type(None) in args:
        # Extract the non-None type
        non_none_args = [t for t in args if t is not type(None)]
        if len(non_none_args) == 1:
            arg_type = non_none_args[0]
            origin = get_origin(arg_type)
            args = get_args(arg_type)

    # 4. Handle Specific Types
    
    # --- Boolean ---
    if arg_type is bool:
        kwargs["type"] = _string_to_bool
        kwargs["nargs"] = "?"
        kwargs["const"] = True
        
        if default_val is not MISSING:
            kwargs["default"] = default_val
        else:
            # Default to False if required=False and no default provided (Standard argparse behavior)
            if not arg.required:
                kwargs["default"] = False

        parser.add_argument(*flags, **kwargs)

        # Add --no-xxx flag if default is True
        if default_val is True:
            no_flags = [f"--no-{f.lstrip('-')}" for f in flags if f.startswith("--")]
            if no_flags:
                parser.add_argument(
                    *no_flags,
                    action="store_false",
                    dest=path,
                    help=f"Disable {path}",
                )

    # --- List / Sequence ---
    elif origin in (list, tuple) or arg_type in (list, tuple):
        kwargs["nargs"] = "+" if arg.nargs is None else arg.nargs
        kwargs["type"] = args[0] if args else str  # Default to str list if generic
        if default_val is not MISSING:
            kwargs["default"] = default_val
        elif arg.required:
            kwargs["required"] = True
        parser.add_argument(*flags, **kwargs)

    # --- Literal / Enum (Choices) ---
    elif origin is Literal:
        kwargs["choices"] = args
        kwargs["type"] = type(args[0])
        if default_val is not MISSING:
            kwargs["default"] = default_val
        elif arg.required:
            kwargs["required"] = True
        parser.add_argument(*flags, **kwargs)

    elif isinstance(arg_type, type) and issubclass(arg_type, Enum):
        kwargs["choices"] = [e.value for e in arg_type]
        kwargs["type"] = type(list(kwargs["choices"])[0])
        if default_val is not MISSING:
            kwargs["default"] = default_val.value if isinstance(default_val, Enum) else default_val
        elif arg.required:
            kwargs["required"] = True
        parser.add_argument(*flags, **kwargs)

    # --- Standard Types ---
    else:
        kwargs["type"] = arg_type
        if arg.nargs is not None:
            kwargs["nargs"] = arg.nargs
        
        if default_val is not MISSING:
            kwargs["default"] = default_val
        elif arg.required:
            kwargs["required"] = True
            
        if arg.choices is not None:
            kwargs["choices"] = arg.choices

        parser.add_argument(*flags, **kwargs)


def populate_parser(parser: argparse.ArgumentParser, refs: Iterable[Ref]) -> None:
    """
    Populate an existing ArgumentParser with arguments defined in the provided Refs.
    """
    for ref in refs:
        if ARG in ref.metadata:
            arg_def = ref.metadata[ARG]
            if not isinstance(arg_def, Arg):
                raise TypeError(
                    f"Invalid metadata for {ARG} in Ref '{ref.path}'. Expected Arg, got {type(arg_def)}."
                )
            _add_argument_for_ref(parser, ref, arg_def)


def parse_refs(
    refs: Iterable[Ref],
    args: Optional[List[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
    return_remaining: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[str]]]:
    """
    Main entry point for parsing arguments based on a list of Refs.
    
    Args:
        refs: A list/iterable of Ref objects defining the available arguments.
        args: Command line arguments to parse (defaults to sys.argv[1:]).
        parser: Optional existing parser to extend.
        return_remaining: If True, returns a tuple (parsed_dict, remaining_args).
    
    Returns:
        A dictionary mapping Ref paths to their parsed values.
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    
    populate_parser(parser, refs)
    
    if args is None:
        args = sys.argv[1:]
        
    namespace, remaining = parser.parse_known_args(args)
    
    # Convert Namespace to Dict
    # Note: We use vars() which handles keys with dots correctly if they were set as dest
    parsed_dict = vars(namespace)
    
    # Filter out keys that might be in the parser but not in our refs (if parser was pre-filled)
    # Actually, we usually want all parsed args.
    
    if return_remaining:
        return parsed_dict, remaining
    
    if remaining:
        # If strict parsing is desired (default behavior of parse_args vs parse_known_args),
        # we should probably raise unless user asked for remaining.
        # But here we mimic the flexible behavior, or we can choose to raise.
        # Standard argparse parse_args raises on unknown. Let's try to mimic that
        # if user didn't ask for remaining.
        parser.parse_args(args) # This will print usage and exit if unknown args exist
        
    return parsed_dict
