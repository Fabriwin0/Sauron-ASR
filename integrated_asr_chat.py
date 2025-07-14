import os
import spaces
import torch
from transformers import pipeline, WhisperTokenizer
import torchaudio
import gradio as gr
from bambara_utils import BambaraWhisperTokenizer
from conversation_agent import ConversationAgent
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine the appropriate device
device = "cpu"  # Change to "cuda" if GPU available

# ASR Model setup
model_checkpoint = "openai/whisper-small"
revision = None
language = "english"

# Load tokenizer and ASR pipeline
tokenizer = WhisperTokenizer.from_pretrained(model_checkpoint, language=language, device=device)
asr_pipe = pipeline("automatic-speech-recognition", model=model_checkpoint, tokenizer=tokenizer, device=device, revision=revision)

# Initialize conversation agent
conversation_agent = ConversationAgent()

def resample_audio(audio_path, target_sample_rate=16000):
    """
    Converts the audio file to the target sampling rate (16000 Hz).
    """
    waveform, original_sample_rate = torchaudio.load(audio_path)
    
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
    return waveform, target_sample_rate

@spaces.GPU()
def process_audio_and_respond(audio, task_type, language, include_analysis=False):
    """
    Complete pipeline: transcribe audio and generate conversational response.
    
    Args:
        audio: Audio file path
        task_type: ASR task (transcribe/translate)
        language: Language for ASR
        include_analysis: Whether to include transcript analysis in output
    
    Returns:
        Dictionary with transcription, response, and optional analysis
    """
    try:
        # Step 1: Transcribe audio
        logger.info("Starting audio transcription...")
        waveform, sample_rate = resample_audio(audio)
        sample = {"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate}
        
        # Get transcription
        transcription_result = asr_pipe(sample, generate_kwargs={"task": task_type, "language": language})
        transcript = transcription_result["text"].strip()
        
        logger.info(f"Transcription completed: {transcript[:50]}...")
        
        # Step 2: Generate conversational response
        logger.info("Generating conversational response...")
        response = conversation_agent.generate_response(transcript)
        
        # Step 3: Prepare output
        result = {
            "transcript": transcript,
            "response": response,
            "conversation_summary": conversation_agent.get_conversation_summary()
        }
        
        if include_analysis:
            analysis = conversation_agent.analyze_transcript(transcript)
            result["analysis"] = analysis
        
        logger.info("Processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in processing pipeline: {e}")
        return {
            "transcript": "Error in transcription",
            "response": "I'm sorry, I encountered an error processing your audio. Please try again.",
            "error": str(e)
        }

def format_output(result):
    """Format the result for display in Gradio interface."""
    if "error" in result:
        return f"❌ Error: {result['error']}", "", ""
    
    transcript = f"🎤 **Transcript:** {result['transcript']}"
    response = f"🤖 **Response:** {result['response']}"
    
    # Format conversation summary
    summary = result.get('conversation_summary', {})
    summary_text = f"""📊 **Conversation Summary:**
- Total exchanges: {summary.get('total_exchanges', 0)}
- Recent topics: {', '.join(summary.get('recent_topics', [])) or 'None detected'}"""
    
    return transcript, response, summary_text

def clear_conversation():
    """Clear the conversation history."""
    conversation_agent.clear_history()
    return "🔄 Conversation history cleared!"

def get_wav_files(directory):
    """Returns a list of .wav files for examples."""
    if not os.path.exists(directory):
        return []
    
    files = os.listdir(directory)
    wav_files = [os.path.abspath(os.path.join(directory, file)) for file in files if file.endswith('.wav')]
    return [[f, "transcribe", "english"] for f in wav_files]

def main():
    """Main function to setup and launch Gradio interface."""
    
    # Get example files
    example_files = get_wav_files("./examples")
    
    # Create Gradio interface
    with gr.Blocks(title="Sauron ASR + Conversational AI", theme=gr.themes.Soft()) as iface:
        gr.Markdown("""
        # 🎙️ Sauron ASR + Conversational AI
        
        **Complete Speech-to-Response Pipeline:**
        1. 🎤 Upload or record audio
        2. 📝 Automatic speech recognition (Whisper)
        3. 🤖 Intelligent response generation (SmolLM)
        4. 💬 Contextual conversation tracking
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### 🎤 Audio Input")
                audio_input = gr.Audio(
                    type="filepath", 
                    label="Upload or Record Audio",
                    value=example_files[0][0] if example_files else None
                )
                
                with gr.Row():
                    task_type = gr.Radio(
                        choices=["transcribe", "translate"], 
                        label="ASR Task", 
                        value="transcribe"
                    )
                    language = gr.Radio(
                        choices=["Russian", "French", "Italian", "Portuguese", "english", "spanish"], 
                        label="Language", 
                        value="english"
                    )
                
                # Control buttons
                with gr.Row():
                    process_btn = gr.Button("🚀 Process Audio", variant="primary", size="lg")
                    clear_btn = gr.Button("🔄 Clear History", variant="secondary")
            
            with gr.Column(scale=3):
                # Output section
                gr.Markdown("### 📋 Results")
                transcript_output = gr.Markdown(label="Transcript")
                response_output = gr.Markdown(label="AI Response")
                summary_output = gr.Markdown(label="Conversation Summary")
                clear_status = gr.Markdown(visible=False)
        
        # Examples section
        if example_files:
            gr.Markdown("### 🎵 Example Audio Files")
            gr.Examples(
                examples=example_files,
                inputs=[audio_input, task_type, language],
                outputs=[transcript_output, response_output, summary_output],
                fn=lambda audio, task, lang: format_output(process_audio_and_respond(audio, task, lang)),
                cache_examples=False
            )
        
        # Event handlers
        process_btn.click(
            fn=lambda audio, task, lang: format_output(process_audio_and_respond(audio, task, lang)),
            inputs=[audio_input, task_type, language],
            outputs=[transcript_output, response_output, summary_output]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=clear_status
        )
        
        # Additional info
        gr.Markdown("""
        ### ℹ️ How it works:
        1. **Speech Recognition**: Uses OpenAI Whisper for accurate transcription
        2. **Context Analysis**: Analyzes sentiment, topics, and conversation flow
        3. **Response Generation**: Uses SmolLM-360M for natural, contextual responses
        4. **Memory**: Maintains conversation history for coherent multi-turn dialogue
        
        ### 🔧 Features:
        - Multi-language support
        - Real-time processing
        - Conversation memory
        - Sentiment analysis
        - Topic detection
        """)
    
    # Launch interface
    iface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()