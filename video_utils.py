import cv2
import os
import json
import re
from pytubefix import YouTube
from pytubefix.cli import on_progress


def dowload_youtube_video(url, save_dir="data/videos"):
    yt = YouTube(url, on_progress_callback = on_progress)
    print("Dowloading: ", yt.title)
    
    ys = yt.streams.get_highest_resolution()
    default_name = ys.get_file_path().split("/")[-1].split('.')[0]
    extension = ys.get_file_path().split("/")[-1].split('.')[1]

    # remove special characters
    default_name = re.sub(r'[^A-Za-z0-9\s]', '', default_name)
    file_name = default_name.lower().replace(" ", "_")+f".{extension}"
    
    saved_path = ys.download(output_path=save_dir, filename=file_name)

    return saved_path


def split_video(video_path, time_step=5, output_dir='data/scenes'):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to skip to capture one frame every n seconds
    frame_skip = int(fps * time_step)

    frame_number = 0
    metadata = {}
    
    video_name = video_path.split("/")[-1].split(".")[0]
    scenes_dir = f'{output_dir}/{video_name}_ts{time_step}'
    metada_dir = output_dir.replace("scenes", "metadatas")
    
    os.makedirs(scenes_dir, exist_ok=True)
    os.makedirs(metada_dir, exist_ok=True)

    metadata_path = f'{metada_dir}/{video_name}_ts{time_step}.json'
    
    if os.path.isfile(metadata_path):
        return metadata_path
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # Calculate the current timestamp in seconds
        timestamp = frame_number / fps

        if frame_number % frame_skip == 0:
            # Save the frame as an image
            frame_filename = f'{scenes_dir}/frame_{frame_number:05d}.jpg'
            
            cv2.imwrite(frame_filename, frame)
            
            # Save the timestamp in the metadata dictionary
            metadata[frame_filename] = timestamp
            print(f'Saved {frame_filename} at {timestamp:.2f} seconds')

        frame_number += 1
        

    # Release the video capture object
    cap.release()

    with open(metadata_path, 'w') as f:
        f.write(json.dumps(metadata))
        
    return metadata_path


if __name__ == "__main__":
    import fire
    fire.Fire()