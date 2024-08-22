import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# llama index ascyncio config
import nest_asyncio
nest_asyncio.apply()

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core import (
    SimpleDirectoryReader, Settings, StorageContext, 
    VectorStoreIndex, QueryBundle
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
import json
import chromadb


def get_transcipt_index(
    transcript_path, 
    reindexing=False,
    segs_per_chunk=2
):
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    file_name = transcript_path.split('/')[-1].split('.')[0]
    coll_name = f'{file_name}_transcript_spc_{segs_per_chunk}'

    # check exist collection
    colls = chroma_client.list_collections()
    coll_names = [coll.name for coll in colls]

    # Init chromadb 
    if coll_name in coll_names:
        vector_collection = chroma_client.get_collection(coll_name)
    else:
        vector_collection = chroma_client.create_collection(coll_name)

    vector_store = ChromaVectorStore(
        chroma_collection=vector_collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print('reindexing', reindexing)
    if coll_name in coll_names and not reindexing:
        print("Loading index from vector store")
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            show_progress=True
        )
        
    else:
        print("Creating index from documents store")
        transcript = json.load(open(transcript_path, 'r'))
        all_segments = transcript['segments']
        all_nodes = []
        for idx in range(0, len(all_segments), segs_per_chunk):
            segments = all_segments[idx:idx+segs_per_chunk]
            text = ' '.join([segment['text'] for segment in segments])
            metadata = {
                'start': segments[0]['start'], 
                'end': sum([segment['end'] for segment in segments]),
                'content type': "spoken content"
            }
            
            node = TextNode(text=text, extra_info=metadata)
            all_nodes.append(node)

        # create vector index
        vector_index = VectorStoreIndex(
            all_nodes,
            storage_context=storage_context, 
            show_progress=True
        )
        
    return vector_index


def get_scene_index(video_descriptions_path, reindexing=False):
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

    file_name = video_descriptions_path.split('/')[-1].split('.')[0]
    coll_name = f'{file_name}_scenes_description'
    
    # init chromadb
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    col_name = video_descriptions_path.split("/")[-1].split('.')[0]

    # check exist collection
    colls = chroma_client.list_collections()
    coll_names = [coll.name for coll in colls]

    # Init chromadb 
    if coll_name in coll_names:
        vector_collection = chroma_client.get_collection(coll_name)
    else:
        vector_collection = chroma_client.create_collection(coll_name)

    vector_store = ChromaVectorStore(
        chroma_collection=vector_collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
    # start indexing
    if coll_name in coll_names and not reindexing:
        print("Creating index from vector store")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            show_progress=True,
        )
    else:
        # parsing nodes with metadata
        all_nodes = []
        video_descriptions = json.load(open(video_descriptions_path, 'r'))
        for scene_desc in video_descriptions:
            desciption = scene_desc.get('desciption')
            metadata = {
                'start': scene_desc['start'], 
                'end': scene_desc['end'],
                'content type': "scene description" 
            }
            node = TextNode(text=desciption, extra_info=metadata)
            all_nodes.append(node)
        
        print("Creating index from nodes")
        index = VectorStoreIndex(
            all_nodes,
            storage_context=storage_context,
            show_progress=True,
        )

    return index

def test_scene_index(video_descriptions_path, reindexing=False):
    # indexing
    scene_index = get_scene_index(video_descriptions_path, reindexing)
    
    query = '''winning scene'''
    query_bundle = QueryBundle(query)
    
    retriever = VectorIndexRetriever(
        index=scene_index,
        similarity_top_k=3,
    )

    retrieved_nodes = retriever.retrieve(query_bundle)
    print("Question:", query)
    print("Retrieved nodes:")
    for idx, node in enumerate(retrieved_nodes):
        print(node.text)
        print(json.dumps(node.metadata, indent=2))
        print("=="*40)

def test_transcipt_index(transcript_path, reindexing=False, spc=2):
    # indexin
    transcipt_index = get_transcipt_index(transcript_path, reindexing=reindexing, segs_per_chunk=spc)
    
    query = '''what is the prize'''
    query_bundle = QueryBundle(query)
    
    retriever = VectorIndexRetriever(
        index=transcipt_index,
        similarity_top_k=3,
    )

    retrieved_nodes = retriever.retrieve(query_bundle)
    print("Question:", query)
    print("Retrieved nodes:")
    for idx, node in enumerate(retrieved_nodes):
        print(node.text)
        print(json.dumps(node.metadata, indent=2))
        print("=="*40)


if __name__ == "__main__":
    import fire
    fire.Fire()