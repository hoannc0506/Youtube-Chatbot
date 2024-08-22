import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import Settings
from llama_index.core.callbacks import (
    CallbackManager, TokenCountingHandler,
    LlamaDebugHandler, CBEventType, CBEvent
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import tiktoken

def init_openai_models(
    model_name="gpt-4o-mini",
    embed_name="text-embedding-ada-002"
):
    '''
        Load models and setup callback handler
    '''
    Settings.llm = OpenAI(model=model_name)
    Settings.embed_model = OpenAIEmbedding(model=embed_name)

    # callback setup
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(model_name).encode,
        verbose=True
    )
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([token_counter, llama_debug])

    return token_counter