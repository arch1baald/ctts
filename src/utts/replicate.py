import os
from typing import Any, Dict

import replicate


def run(
    api_key: str,
    model_path: str,
    input_data: Dict[str, Any],
) -> bytes:
    """
    Run a model on Replicate and get the result as bytes.

    Args:
        model_path: The model path in format "username/model-name"
        input_data: Dictionary of input parameters for the model

    Returns:
        Generated audio data as bytes
    """
    os.environ["REPLICATE_API_TOKEN"] = api_key
    output = replicate.run(model_path, input=input_data)

    # Handle different output types
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], str) and output[0].startswith("http"):
        # Download the audio file from URL
        import requests

        response = requests.get(output[0])
        return response.content
    elif isinstance(output, bytes):
        return output
    elif hasattr(output, "__str__") and str(output).startswith("http"):
        # Handle replicate.helpers.FileOutput or similar objects that have URL as string representation
        import requests

        url = str(output)
        response = requests.get(url)
        return response.content
    else:
        raise ValueError(f"Unexpected output type: {type(output)}, content: {output}")
