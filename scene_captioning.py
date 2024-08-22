from openai import OpenAI
import os
import base64
from PIL import Image
from io import BytesIO
import json
import time

def image_to_base64(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to a byte array
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

    # Encode the byte array to base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def describe_scene(img_path):
    client = OpenAI()
    base64_image = image_to_base64(img_path)

    describe_system_prompt = '''\
You are a system generating description for scene from video to be used on an entertainment platform. \
You will describe the scene captured in the video, giving details but staying concise. \
Focus on the key elements of the scene, including the setting, characters, actions, and any significant objects or events. \
If the scene has a specific mood, atmosphere, or style, briefly mention it.'''
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    messages=[
        {
            "role": "system",
            "content": describe_system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
            ],
        }
    ],
    max_tokens=300,
    )

    return response.choices[0].message.content


def descipe_video_scenes(metadata_path, out_dir="data/desciptions", debug=False):
    # if not out_file:
    file_name = metadata_path.split("/")[-1]

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, file_name)
    
    video_metadata = json.load(open(metadata_path, 'r'))

    video_desciptions = []

    print("Generating scene description")
    t_start = time.time()
    for idx, (image_path, time_stamp) in enumerate(video_metadata.items()):
        scene_dict = {}
        # print(time_stamp)
        scene_dict['start'] = int(time_stamp)
        scene_dict['end'] = scene_dict['start'] + 1
        scene_dict['file_path'] = image_path

        # get scene description using openai api
        scene_dict['desciption'] = describe_scene(image_path)
        
        print(scene_dict)
        video_desciptions.append(scene_dict)

        # debug
        if debug and idx == 2:
            break

    t_end = time.time()
    print(f"Generated scenes descriptions in:{t_end - t_start:.2f}s")
    with open(out_file, 'w') as f:
        f.write(json.dumps(video_desciptions))

    print("Saved video scenes descriptions to", out_file)
    
    return out_file

if __name__ == "__main__":
    import fire
    fire.Fire(descipe_video_scenes)