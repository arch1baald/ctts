import asyncio
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Type, TypeVar, Union

import pandas as pd
from IPython.display import HTML, Audio, display

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

    print(f"Started {len(_tasks)} tasks...")

    results = []
    for future in asyncio.as_completed(_tasks):
        result = await future

        # Format result data
        func_name = result["func"].__name__ if hasattr(result["func"], "__name__") else str(result["func"])
        module_name = result["func"].__module__ if hasattr(result["func"], "__module__") else ""
        full_name = f"{module_name}.{func_name}" if module_name else func_name

        # Create data for the transposed table
        table_data = {
            "Model": full_name,
            "Text": result["text"][:200] + ("..." if len(result["text"]) > 200 else ""),
            "Time (s)": f"{result['elapsed']:.2f}",
            "Status": "Success" if result["success"] else "Failed",
        }

        # Add all parameters to the table
        for k, v in result["params"].items():
            # Limit the length of parameter values for display
            v_str = str(v)
            table_data[f"Param: {k}"] = v_str[:100] + ("..." if len(v_str) > 100 else "")

        # Create a transposed DataFrame (columns become rows)
        df = pd.DataFrame.from_dict(table_data, orient="index", columns=["Value"])  # type: ignore

        display(HTML(f"<h3>Result {len(results) + 1}</h3>"))
        table_html = df.to_html()
        display(HTML(table_html))

        # Display audio if available
        if result["success"]:
            audio_data = result["result"]
            display(Audio(audio_data))
        else:
            print(result["error"])

        # Add separator line
        display(HTML("<hr style='margin: 2em 0;'>"))

        results.append(result)

    print("Done!")
    return results
