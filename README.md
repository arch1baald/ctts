# CTTS
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch1baald/ctts/blob/main/example.ipynb?target=_blank)

Compare different text-to-speech models:
- [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech)
- [ElevenLabs](https://elevenlabs.io/)
- [Kokoro](https://replicate.com/cjwbw/kokoro)
- [Orpheus](https://replicate.com/scuffedcontent/orpheus-v1)
- [Zyphra/Zonos](https://playground.zyphra.com/audio)
- [Hume AI](https://dev.hume.ai/docs/text-to-speech-tts/quickstart/python)
- [Cartesia](https://docs.cartesia.ai/)


## Installation

```bash
pip install --upgrade git+https://github.com/arch1baald/ctts.git
```

## Quick Start

The simplest way to get started is to open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch1baald/ctts/blob/main/example.ipynb?target=_blank)


From Jupyter:
```python
import os
from IPython.display import Audio
import ctts.openai

os.environ["OPENAI__API_KEY"] = "<openai-api-key>"
audio = ctts.openai.generate('Hello, world!', 'echo')
Audio(audio)
```

## Development

### Prerequisites

- Python 3.11.12 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Modern Python package installer and resolver
- Make - for running development commands

### Setup

Clone the repository:
```bash
git clone https://github.com/arch1baald/ctts.git
cd ctts
```

Install in development mode:
```bash
make install-dev
```

### Development Commands

For all available commands:
```bash
make help
```

Run linting and type checking:
```bash
make lint
```

Format code:
```bash
make format
```
