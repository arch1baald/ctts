# CTTS

CTTS is a tool for comparing text-to-speech models.

## Installation

### Prerequisites

- Python 3.11.12 or higher
- uv - An extremely fast Python package and project manager, written in Rust. https://docs.astral.sh/uv/getting-started/installation/

### Installing the Package

To install the package locally:

```bash
make install
```

To install the package in Google Colab:

```python
!pip install --upgrade git+https://github.com/arch1baald/ctts.git
```

For development, install in editable mode with pre-commit hooks:

```bash
make install-dev
```

## Quick Start

From Jupyter:

```python
import os
from IPython.display import Audio
import ctts.openai

os.environ["OPENAI__API_KEY"] = "<openai-api-key>"
audio = ctts.openai.generate('Hello, world!', 'echo')
Audio(audio)
```
