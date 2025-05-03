from enum import Enum
from typing import Any, Type, TypeVar

T = TypeVar("T", bound=Enum)


def convert_to_enum(enum_class: Type[T], value: Any) -> T:
    if isinstance(value, enum_class):
        return value

    try:
        return enum_class(value)
    except ValueError:
        valid_values = [str(v.value) for v in enum_class]
        raise ValueError(
            f"Invalid value: '{value}' for {enum_class.__name__}. Valid options are: {', '.join(valid_values)}"
        )
