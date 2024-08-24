import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# import nest_asyncio
# nest_asyncio.apply()

import llama_index.core
from llama_index.core.callbacks import (
    CallbackManager, TokenCountingHandler, LlamaDebugHandler
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
import tiktoken
import index_utils, query_utils

def set_logging_handlers(
    model_name="gpt-4o-mini"
):
    # callback setup
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(model_name).encode,
        verbose=True
    )
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    llama_index.core.set_global_handler("simple")
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([token_counter, llama_debug])    



def get_agent(
    transcript_path="./data/audio_transcripts/in_10_minutes_this_room_will_explode.json",
    video_descriptions_path="./data/desciptions/in_10_minutes_this_room_will_explode_ts5.json",
    verbose=False
):
    
    # Load models and setup callback handler
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    
    # load video indices
    transcript_index = index_utils.get_transcipt_index(
        transcript_path="./data/audio_transcripts/in_10_minutes_this_room_will_explode.json",
        segs_per_chunk=1
    )

    scene_index = index_utils.get_scene_index(
        video_descriptions_path="./data/desciptions/in_10_minutes_this_room_will_explode_ts5.json"
    )

    # init agent tools
    agent_tools = query_utils.get_tools(transcript_index, scene_index)
    
    agent_context = '''Your are video analysis expert, you can answer question related to poken content or visual scene descriptions.\
Focus on accurately understanding the context, intent, and nuances of the spoken content and scene description content. \
If a query asks for details not mentioned in the content, indicate that the information is not available.'''
    
    react_agent = ReActAgent.from_tools(
        agent_tools,
        context=agent_context,
        verbose=verbose
    )
    
    return react_agent


def sample_chatbot():
    agent = get_agent()
    while True:
        text_input = input("User: ")
        if text_input == "exit":
            break
        response = agent.chat(text_input)
        print(f"Response\n: {response}")


def test_agent(debugging=False):
    if debugging:
        set_logging_handlers()
        
    agent = get_agent()
    queries = [
        "what is the prize",
        "Describe the winning scene",
        "What happened in the first 3 minutes of the video?"
    ]

    for query in queries:
        print(f"Question\n: {query}")
        response = agent.chat(query)
        print(f"Response\n: {response}")


if __name__ == "__main__":
    import fire
    fire.Fire()