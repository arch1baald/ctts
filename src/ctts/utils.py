import asyncio
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Type, TypeVar, Union

from IPython.display import Audio, display

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


async def run_task(
    func: Callable[[str], Awaitable[Any]], text: str, params: Dict[str, Any], index: int = 0
) -> Dict[str, Any]:
    """
    Asynchronously runs a generation function and returns a dictionary with the result.

    Args:
        func: The function to run
        text: The text to pass to the function
        params: The parameters to pass to the function
        index: The index of the task

    Returns:
        A dictionary with the result of the function
    """
    start_time = time.time()
    try:
        result = await func(text, **params)
        elapsed = time.time() - start_time
        return {
            "index": index,
            "func": func,
            "text": text,
            "params": params,
            "result": result,
            "elapsed": elapsed,
            "success": True,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "index": index,
            "func": func,
            "text": text,
            "params": params,
            "error": str(e),
            "elapsed": elapsed,
            "success": False,
        }


async def batch_agenerate(
    tasks: List[
        Union[Tuple[Callable[[str], Awaitable[Any]], str], Tuple[Callable[[str], Awaitable[Any]], str, Dict[str, Any]]]
    ],
) -> List[Dict[str, Any]]:
    """
    Asynchronously runs multiple generation functions and displays results as they complete.

    Args:
        tasks: List of tuples in format [(function1, text1), (function2, text2, params2), ...]
              If params are not provided, empty dict will be used as default

    Returns:
        List of results from each function in order of completion
    """

    _tasks = []
    for i, task_tuple in enumerate(tasks):
        if len(task_tuple) == 2:
            func, text = task_tuple
            params = {}
        elif len(task_tuple) == 3:
            func, text, params = task_tuple
        else:
            raise ValueError(f"Invalid task tuple length: {len(task_tuple)}. Expected 2 or 3 elements.")

        _tasks.append(run_task(func, text, params, i))

    results = []
    for future in asyncio.as_completed(_tasks):
        result = await future

        display((result["func"], result["text"], result["params"], result["elapsed"]))
        if result["success"]:
            display(Audio(result["result"]))
        else:
            display(result["error"])
        results.append(result)

    return results
