"""
Argument parsing utilities for slyme, designed to work with Ref metadata.
Refactored to separate argument definition/collection from parsing/injection.
"""

import argparse
import sys
from dataclasses import replace
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Union,
    Type,
    get_origin,
    get_args,
    Literal,
    Dict,
)
from slyme.context import Ref, Context
from slyme.node.core import NODE_PYTREE_ENGINE
from deepslyme.context.metadata import ARG, HELP, TYPE, Arg

__all__ = [
    "ARG",
    "Arg",
    "collect_refs",
    "resolve_args",
    "populate_parser",
    "parse_and_inject",
]


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


def _infer_type(arg: Arg) -> Type:
    """
    Infer the type of the argument based on explicit definition or default value.
    """
    if arg.type is not None:
        return arg.type

    default_val = arg.resolve_default()
    if not Arg.is_missing(default_val) and default_val is not None:
        return type(default_val)

    # Fallback: if no type info is available, assume str
    return str


# --- 1. Collection & Resolution ---


def collect_refs(element: Any) -> List[Ref]:
    """
    Collect all Ref objects from a node structure (NodeDef, NodeExec, etc.)
    using the NODE_PYTREE_ENGINE.
    """
    refs: List[Ref] = []

    def is_leaf(node: Any, _) -> bool:
        return isinstance(node, Ref)

    # We iterate using NODE_PYTREE_ENGINE which knows how to traverse Node structures
    for _, leaf in NODE_PYTREE_ENGINE.iter_with_path(element, is_leaf=is_leaf):
        if isinstance(leaf, Ref):
            refs.append(leaf)

    return refs


def resolve_args(refs: Iterable[Ref]) -> Dict[str, Arg]:
    """
    Resolve a list of Refs into a mapping of {path: Arg}.

    Handles conflicts:
    - If multiple Refs point to the same path, they must have compatible Arg definitions.
    - Enforces strict equality for Arg definitions if they exist.

    Handles merging:
    - If Arg is missing help/type, tries to fill it from metadata[HELP] / metadata[TYPE].
    """
    path_to_args: Dict[str, List[Arg]] = {}
    path_to_help: Dict[str, List[str]] = {}
    path_to_type: Dict[str, List[Any]] = {}

    for ref in refs:
        if ARG in ref.metadata:
            arg_def = ref.metadata[ARG]
            if not isinstance(arg_def, Arg):
                raise TypeError(
                    f"Invalid metadata for {ARG} in Ref '{ref.path}'. Expected Arg, got {type(arg_def)}."
                )
            path_to_args.setdefault(ref.path, []).append(arg_def)

        if HELP in ref.metadata:
            path_to_help.setdefault(ref.path, []).append(ref.metadata[HELP])

        if TYPE in ref.metadata:
            path_to_type.setdefault(ref.path, []).append(ref.metadata[TYPE])

    resolved: Dict[str, Arg] = {}

    for path, args in path_to_args.items():
        # 1. Resolve Arg Conflict (Strict Equality)
        base_arg = args[0]
        for other in args[1:]:
            if base_arg != other:
                raise ValueError(
                    f"Conflicting Arg definitions for path '{path}':\n"
                    f"1. {base_arg}\n"
                    f"2. {other}\n"
                    "Ensure all Refs for the same path use the same Arg definition."
                )

        # 2. Merge with HELP/TYPE (if needed)
        # Handle immutable Arg by collecting changes first
        changes = {}
        if base_arg.help is None and path in path_to_help:
            # Use the first available help string
            changes["help"] = path_to_help[path][0]

        if base_arg.type is None and path in path_to_type:
            # Use the first available type
            changes["type"] = path_to_type[path][0]

        if changes:
            final_arg = replace(base_arg, **changes)
        else:
            final_arg = base_arg

        resolved[path] = final_arg

    return resolved


# --- 2. Parsing & Injection ---


def _add_argument(parser: argparse.ArgumentParser, path: str, arg: Arg):
    """
    Add a single argument to the parser.
    """
    # 1. Determine Flag Names
    # Convert dotted path "model.config.lr" -> "--model.config.lr" and "--model-config-lr"
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
        "dest": path,  # Keep the dot notation for the destination key
        "help": arg.help,
        "metavar": arg.metavar,
    }

    # 3. Resolve Type and Default
    arg_type = _infer_type(arg)
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

        # Handle default values
        if not Arg.is_missing(default_val):
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
        if not Arg.is_missing(default_val):
            kwargs["default"] = default_val
        elif arg.required:
            kwargs["required"] = True
        parser.add_argument(*flags, **kwargs)

    # --- Literal / Enum (Choices) ---
    elif origin is Literal:
        kwargs["choices"] = args
        kwargs["type"] = type(args[0])
        if not Arg.is_missing(default_val):
            kwargs["default"] = default_val
        elif arg.required:
            kwargs["required"] = True
        parser.add_argument(*flags, **kwargs)

    elif isinstance(arg_type, type) and issubclass(arg_type, Enum):
        kwargs["choices"] = [e.value for e in arg_type]
        kwargs["type"] = type(list(kwargs["choices"])[0])
        if not Arg.is_missing(default_val):
            kwargs["default"] = (
                default_val.value if isinstance(default_val, Enum) else default_val
            )
        elif arg.required:
            kwargs["required"] = True
        parser.add_argument(*flags, **kwargs)

    # --- Standard Types ---
    else:
        kwargs["type"] = arg_type
        if arg.nargs is not None:
            kwargs["nargs"] = arg.nargs

        if not Arg.is_missing(default_val):
            kwargs["default"] = default_val
        elif arg.required:
            kwargs["required"] = True

        if arg.choices is not None:
            kwargs["choices"] = arg.choices

        parser.add_argument(*flags, **kwargs)


def populate_parser(parser: argparse.ArgumentParser, args_map: Dict[str, Arg]) -> None:
    """
    Populate an existing ArgumentParser with resolved arguments.
    """
    for path, arg in args_map.items():
        _add_argument(parser, path, arg)


def parse_and_inject(
    source: Union[Any, Iterable[Ref]],
    context: Optional[Context] = None,
    parser: Optional[argparse.ArgumentParser] = None,
    cli_args: Optional[List[str]] = None,
    custom_args: Optional[Dict[str, Arg]] = None,
) -> Union[Dict[str, Any], Context]:
    """
    High-level entry point to parse arguments and optionally inject them into a Context.

    Args:
        source: Either a Node element (to auto-collect Refs) or an iterable of Refs.
        context: Optional Context to inject parsed values into. If provided, returns a new Context.
        parser: Optional existing parser to extend.
        cli_args: Command line arguments to parse (defaults to sys.argv[1:]).
        custom_args: Additional arguments to add/override, mapping path -> Arg.

    Returns:
        If context is provided: A new Context with injected values.
        If context is None: A dictionary of parsed values.
    """
    # 1. Collect Refs
    if (
        isinstance(source, Iterable)
        and not hasattr(source, "__iter_with_path__")
        and not isinstance(source, (str, bytes))
    ):
        # Rough check for Iterable[Ref], assuming source isn't the Node structure itself (which might be iterable?)
        # NODE structures are usually not directly iterable as Refs.
        # Safer: Check if the first element is Ref?
        # But source could be empty list.
        # Let's rely on type checking or assume if it's a list/tuple of Refs it's Refs.
        # If it's a NodeDef/NodeExec, it's not a list of Refs.
        if isinstance(source, (list, tuple)) and (
            not source or isinstance(source[0], Ref)
        ):
            refs = source
        else:
            # Assume it's a Node structure
            refs = collect_refs(source)
    else:
        refs = collect_refs(source)

    # 2. Resolve Args
    args_map = resolve_args(refs)

    # 3. Merge Custom Args
    if custom_args:
        args_map.update(custom_args)

    # 4. Populate Parser
    if parser is None:
        parser = argparse.ArgumentParser()

    populate_parser(parser, args_map)

    # 5. Parse
    if cli_args is None:
        cli_args = sys.argv[1:]

    namespace = parser.parse_args(cli_args)
    parsed_values = vars(namespace)

    # 6. Inject or Return
    if context is None:
        return parsed_values

    # Prepare updates for Context.mutate
    # parsed_values is { "path": value, ... }
    # Context.mutate expects { Ref: value }
    updates = {}
    for key, value in parsed_values.items():
        # Only inject if it looks like a path (non-empty string)
        if key:
            updates[Ref(key)] = value

    return context.mutate(updates=updates, drops=set())
