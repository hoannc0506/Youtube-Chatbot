import os
import sys
import video_utils, audio_utils
import agent_utils


def video_chatbot(
    video_path_or_url="https://www.youtube.com/watch?v=Pv0iVoSZzN8"
):
    
    if "https://www.youtube.com" in video_path_or_url:
        video_path = video_utils.dowload_youtube_video(video_path_or_url)
    else:
        video_path = video_path_or_url

    # scene processing
    print('Spliting video to scenes')
    scene_data_path = video_utils.split_video(video_path)

    # audio processing
    print('Convert video audio to transcipts')
    audio_path = audio_utils.video_to_audio(video_path)
    transcript_data_path = audio_utils.audio_to_transciption(audio_path)

    # init agent
    print("initializing chatbot")
    chatbot = agent_utils.get_agent(transcript_data_path, scene_data_path)

    return chatbot

if __name__ == "__main__":
    chatbot =  video_chatbot()
    
    while True:
        text_input = input("User: ")
        if text_input == "exit":
            break
        response = chatbot.chat(text_input)
        print(f"Response\n: {response}")
    
    