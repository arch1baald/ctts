# UTTS
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch1baald/utts/blob/main/examples/example.ipynb)

Universal interface to test and compare text-to-speech models. Currently supports:
- [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech)
- [ElevenLabs](https://elevenlabs.io/)
- [Kokoro](https://replicate.com/cjwbw/kokoro)
- [Orpheus](https://replicate.com/scuffedcontent/orpheus-v1)
- [Zyphra/Zonos](https://playground.zyphra.com/audio)
- [Hume AI](https://dev.hume.ai/docs/text-to-speech-tts/quickstart/python)
- [Cartesia](https://docs.cartesia.ai/)


## Installation

```bash
pip install utts
```

or install from source:
```bash
pip install --upgrade git+https://github.com/arch1baald/utts.git
```

Obtain API keys for the services you want to use:
- [OpenAI](https://platform.openai.com/settings/api-keys)
- [ElevenLabs](https://elevenlabs.io/app/settings/api-keys)
- [Replicate](https://replicate.com/account/api-tokens) (for Kokoro and Orpheus)
- [Zyphra/Zonos](https://playground.zyphra.com/settings/api-keys)
- [Hume AI](https://platform.hume.ai/settings/keys)
- [Cartesia](https://play.cartesia.ai/keys)

## Quick Start

The simplest way to get started is to open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch1baald/utts/blob/main/examples/example.ipynb)


Or test from Jupyter Notebook:
```python
import os
from IPython.display import Audio
import utts.openai

os.environ["OPENAI__API_KEY"] = "<openai-api-key>"
audio = utts.openai.generate('Hello, world!', 'echo')
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
git clone https://github.com/arch1baald/utts.git
cd utts
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
