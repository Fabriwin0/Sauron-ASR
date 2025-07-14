# Sauron-ASR

An integrated speech recognition and conversational AI system that combines OpenAI Whisper for transcription with SmolLM for intelligent response generation.

## Features

🎤 **Speech Recognition**: High-quality transcription using OpenAI Whisper
🤖 **Conversational AI**: Natural response generation using SmolLM-360M
💬 **Context Awareness**: Maintains conversation history and context
📊 **Analysis**: Sentiment analysis and topic detection
🌍 **Multi-language**: Support for multiple languages
🎯 **Flexible Interface**: Both web UI and CLI options

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Sauron-ASR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Check installation:
```bash
python run_integrated_system.py --check-deps
```

### Usage

#### Web Interface (Recommended)
Launch the interactive web interface:
```bash
python run_integrated_system.py
```

Then open your browser to `http://localhost:7860`

#### CLI Mode
Process a single audio file:
```bash
python run_integrated_system.py --cli --audio examples/sample.wav
```

With translation:
```bash
python run_integrated_system.py --cli --audio examples/sample.wav --task translate --language spanish
```

#### Testing
Run system tests:
```bash
python run_integrated_system.py --test
```

## System Architecture

```
Audio Input → Whisper ASR → Transcript → SmolLM → Response
                ↓                          ↑
            Analysis ←→ Context Memory ←→ History
```

### Components

1. **ASR Pipeline** (`nofile.py`, `bambara_utils.py`)
   - OpenAI Whisper for speech recognition
   - Support for multiple languages including Bambara
   - Audio preprocessing and resampling

2. **Conversation Agent** (`conversation_agent.py`)
   - SmolLM-360M for response generation
   - Context analysis and sentiment detection
   - Conversation memory management

3. **Integration Layer** (`integrated_asr_chat.py`)
   - Combines ASR and conversation components
   - Gradio web interface
   - Real-time processing pipeline

4. **CLI Interface** (`run_integrated_system.py`)
   - Command-line access
   - Batch processing capabilities
   - System testing and validation

## Configuration

### Model Settings
- **ASR Model**: `openai/whisper-small` (configurable)
- **LLM Model**: `HuggingFaceTB/SmolLM-360M-Instruct`
- **Device**: Auto-detected (CUDA/CPU)

### Conversation Settings
- **History Length**: 5 exchanges (configurable)
- **Response Length**: Max 200 characters
- **Temperature**: 0.7 for balanced creativity

## Supported Languages

- English
- Spanish
- French
- Italian
- Portuguese
- Russian
- Bambara (custom support)

## API Reference

### ConversationAgent

```python
from conversation_agent import ConversationAgent

agent = ConversationAgent()
response = agent.generate_response("Hello, how are you?")
analysis = agent.analyze_transcript("I'm feeling great today!")
summary = agent.get_conversation_summary()
```

### Key Methods

- `generate_response(transcript, context=None)`: Generate conversational response
- `analyze_transcript(transcript)`: Analyze text for sentiment and topics
- `get_conversation_summary()`: Get conversation statistics
- `clear_history()`: Reset conversation memory

## Examples

### Basic Conversation
```python
agent = ConversationAgent()

# First interaction
response1 = agent.generate_response("Hi there!")
print(response1)  # "Hello! How can I help you today?"

# Follow-up with context
response2 = agent.generate_response("What's the weather like?")
print(response2)  # "I don't have access to current weather data, but I'd be happy to help you find a weather service!"
```

### Audio Processing
```python
from integrated_asr_chat import process_audio_and_respond

result = process_audio_and_respond(
    audio="path/to/audio.wav",
    task_type="transcribe",
    language="english"
)

print(f"Transcript: {result['transcript']}")
print(f"Response: {result['response']}")
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure sufficient RAM (2GB+ recommended)
   - Check internet connection for model downloads
   - Try CPU mode if GPU issues occur

2. **Audio Processing Issues**
   - Verify audio file format (WAV recommended)
   - Check sample rate (16kHz preferred)
   - Ensure ffmpeg is installed

3. **Dependencies**
   - Run `python run_integrated_system.py --check-deps`
   - Reinstall with `pip install -r requirements.txt --force-reinstall`

### Performance Tips

- Use GPU for faster processing
- Limit conversation history for memory efficiency
- Use smaller audio files for real-time processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

GNU General Public License v3.0 - see LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Hugging Face for SmolLM and transformers
- Gradio for the web interface
- The open-source community for various tools and libraries
