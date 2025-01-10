import os

import spaces
import torch
from transformers import pipeline, WhisperTokenizer
import torchaudio
import gradio as gr
# Please note that the below import will override whisper LANGUAGES to add bambara
# this is not the best way to do it but at least it works. for more info check the bambara_utils code
from bambara_utils import BambaraWhisperTokenizer

# Determine the appropriate device (GPU or CPU)
device = "cpu" #if torch.cuda.is_available() else "cpu"

# Define the model checkpoint and language
model_checkpoint = "openai/whisper-small"
revision = None
language = "english"  # Default language

# Load the custom tokenizer designed for Bambara and the ASR model
tokenizer = WhisperTokenizer.from_pretrained(model_checkpoint, language=language, device=device)
pipe = pipeline("automatic-speech-recognition", model=model_checkpoint, tokenizer=tokenizer, device=device, revision=revision)


def resample_audio(audio_path, target_sample_rate=16000):
    """
    Converts the audio file to the target sampling rate (16000 Hz).
    
    Args:
        audio_path (str): Path to the audio file.
        target_sample_rate (int): The desired sample rate.

    Returns:
        A tensor containing the resampled audio data and the target sample rate.
    """
    waveform, original_sample_rate = torchaudio.load(audio_path)
    
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
    return waveform, target_sample_rate

@spaces.GPU()
def transcribe(audio, task_type, language):
    """
    Transcribes the provided audio file into text using the configured ASR pipeline.

    Args:
        audio: The path to the audio file to transcribe.
        task_type: The type of task to perform (transcribe or translate).
        language: The language to use for transcription or translation.

    Returns:
        A string representing the transcribed text.
    """
    # Convert the audio to 16000 Hz
    waveform, sample_rate = resample_audio(audio)
    
    # Use the pipeline to perform transcription or translation
    sample = {"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate}
    text = pipe(sample, generate_kwargs={"task": task_type, "language": language})["text"]
    
    return text

def get_wav_files(directory):
    """
    Returns a list of absolute paths to all .wav files in the specified directory.

    Args:
        directory (str): The directory to search for .wav files.

    Returns:
        list: A list of absolute paths to the .wav files.
    """
    # List all files in the directory
    files = os.listdir(directory)
    # Filter for .wav files and create absolute paths
    wav_files = [os.path.abspath(os.path.join(directory, file)) for file in files if file.endswith('.wav')]
    wav_files = [[f, "transcribe"] for f in wav_files]

    return wav_files

def main():
    # Get a list of all .wav files in the examples directory
    example_files = get_wav_files("./examples")

    # Setup Gradio interface
    iface = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(type="filepath", value=example_files[0][0]),
            gr.Radio(choices=["transcribe", "translate"], label="Task Type", value="transcribe"),
            gr.Radio(choices=["Russian", "French", "Italian", "Portuguese", "english", "spanish"], label="Language", value="english")
        ],
        outputs="text",
        title="Sauron ASR",
        description="Realtime demo for speech recognition and translation based on the Whisper model.",
        examples=example_files,
        cache_examples="lazy",
    )

    # Launch the interface
    iface.launch(share=False)


if __name__ == "__main__":
    main()
