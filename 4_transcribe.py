import os
import sys
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: python 4_transcribe.py <video_file_path> <output_text_file_path>")
    sys.exit(1)

video_path = sys.argv[1]
output_text_path = sys.argv[2]
device="cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    sys.exit(1)
output_dir = Path(output_text_path).parent
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
print("Loading faster-whisper model...")
model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device=device,
    compute_type="int8_float16" if device == "cuda" and torch.cuda.is_available() else "int8"
)

batched_model = BatchedInferencePipeline(model=model)
print(f"Transcribing audio from {video_path}...")
try:
    segments, info = batched_model.transcribe(
        video_path, 
        multilingual=True, 
        language="ru", 
        batch_size=8,
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    transcription = " ".join(segment.text for segment in segments).strip()
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(transcription)
    print(f"Transcription completed successfully!")
    print(f"Transcription saved to: {output_text_path}")
    print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
except Exception as e:
    print(f"Error during transcription: {str(e)}")
    sys.exit(1)
