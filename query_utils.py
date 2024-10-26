import logging
import sys
import json
# import nest_asyncio
# nest_asyncio.apply()
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from llama_index.core import (
    Settings,
    VectorStoreIndex, 
    SummaryIndex,
    StorageContext,
    QueryBundle,
    get_response_synthesizer
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import index_utils


def get_tools(transcript_index, scene_index):
    vector_store_info = VectorStoreInfo(
        content_info="Video entertainment content",
        metadata_info=[
            MetadataInfo(
                name="start",
                type="float",
                description="Start time of a shot in seconds"
            ),
            MetadataInfo(
                name="end",
                type="float",
                description="End time of a shot in seconds"
            ),
        ],
    )

    # define retriever with metadata
    similarity_top_k = 10
    rerank_top_n = 5
    transcript_retriever = VectorIndexAutoRetriever(
        transcript_index, 
        vector_store_info=vector_store_info,
        similarity_top_k=similarity_top_k 
    )
    
    scene_retriever = VectorIndexAutoRetriever(
        scene_index, 
        vector_store_info=vector_store_info,
        similarity_top_k=similarity_top_k 
    )

    # init reranker
    rerank_postprocessor = SentenceTransformerRerank(
        model='models/mxbai-rerank-xsmall-v1',
        top_n=rerank_top_n, # number of nodes after re-ranking,
        keep_retrieval_score=False,
        device="cuda:0"
    )
    
    # define response synthesizers
    compact_synthesizer = get_response_synthesizer(response_mode="compact")
    summary_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

    # define query engines
    transcript_query_engine = RetrieverQueryEngine(
        retriever=transcript_retriever,
        response_synthesizer=compact_synthesizer,
        node_postprocessors=[rerank_postprocessor],
    )
    
    scene_query_engine = RetrieverQueryEngine(
        retriever=scene_retriever,
        response_synthesizer=compact_synthesizer,
        node_postprocessors=[rerank_postprocessor],
    )
    
    scene_summary_engine = RetrieverQueryEngine(
        retriever=scene_retriever,
        response_synthesizer=summary_synthesizer,
        # node_postprocessors=[rerank_postprocessor],
    )

    # return query engine only
    # return transcript_query_engine, scene_query_engine, scene_summary_engine

    # define tools
    transcipt_tool = QueryEngineTool(
        query_engine=transcript_query_engine,
        metadata=ToolMetadata(
            name=f"transcript_tool",
            description=(
                '''answer questions related to audio transcripts, conversations, spoken content'''
            )
        )
    )
    
    scene_tool = QueryEngineTool(
        query_engine=scene_query_engine,
        metadata=ToolMetadata(
            name=f"scene_tool",
            description=(
                '''answer questions related to scene description content'''
            )
        )
    )

    summary_tool = QueryEngineTool(
        query_engine=scene_summary_engine,
        metadata=ToolMetadata(
            name=f"scene_summary_tool",
            description=(
                '''summarize content from video scenes descriptions'''
            )
        )
    )

    tools = [transcipt_tool, scene_tool, summary_tool]
    
    return tools




    
    