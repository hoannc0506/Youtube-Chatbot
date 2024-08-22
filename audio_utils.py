from moviepy.editor import VideoFileClip
import os
from openai import OpenAI
import time
import json

def video_to_audio(video_path, save_dir="data/audios"):
    os.makedirs(save_dir, exist_ok=True)
    video_name = video_path.split("/")[-1].split(".")[0]
    saved_path = f"{save_dir}/{video_name}.mp3"
    if os.path.isfile(saved_path):
        return saved_path
    
    # Load the video file
    video = VideoFileClip(video_path)
    
    # Extract the audio
    audio = video.audio
    
    # Save the audio to a file (in mp3 format)
    audio.write_audiofile(saved_path)
    
    # Clean up resources
    video.close()

    return saved_path

    
def audio_to_transciption(audio_path, save_dir="./data/audio_transcripts/"):
    file_name = audio_path.split("/")[-1].split(".")[0]
    save_path = f"{save_dir}/{file_name}.json"
    if os.path.isfile(save_path):
        return save_path
        
    client = OpenAI()
    print("Generating transcription")

    audio_file = open(audio_path, "rb")
    
    t_start = time.time()
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )

    
    os.makedirs(save_dir, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write(json.dumps(transcript.to_dict(), indent=2))

    t_end = time.time()
    print(f"Generated audio description in:{t_end - t_start:.2f}s")
    print("Saved transcript to:", save_path)
    
    return save_path
        
if __name__ == "__main__":
    import fire
    fire.Fire()