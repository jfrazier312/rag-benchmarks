import os
from dotenv import load_dotenv

from ingestion.load import load_dataset, load_from_url
from ingestion.chunking import token_text_split, recursive_character_text_split
from ingestion.embeddings import open_ai_embeddings
from ingestion.storage.astradb import initialize_astra_db
from retrieval.chains import as_retriever, basic_chat, basic_chat_with_memory
from retrieval.prompts import PHILOSOPHER_PROMPT
from generation.models import chat_open_ai
from generation.query_loop import query_loop

from langchain_core.documents import Document


def load_data():
    documents = load_from_url(
        "https://raw.githubusercontent.com/CassioML/cassio-website/main/docs/frameworks/langchain/texts/amontillado.txt",
        "data/amontillado.txt",
    )
    return documents


def split(documents):
    split_documents = token_text_split(documents, chunk_size=512, chunk_overlap=64)
    return split_documents


def prompt():
    prompt = """
    You are a very smart and helpful assistant that only knows about the provided context. Do not answer
    any questions that are not related to the context. Answer with extreme detail, pulling 
    quotes and supporting context directly from the provided context.  
    """
    return prompt


def retrieval_chain(retriever, model, prompt):
    chain = basic_chat_with_memory(retriever, model, prompt)
    return chain


def rag_starter_app():
    # Initialize environment variables
    load_dotenv()
    astra_db_token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    # Ingestion
    documents = load_data()

    # Chunking
    split_documents = split(documents)

    # Storage / Embedding
    collection = input("Collection: ")
    embedding = open_ai_embeddings()
    vstore = initialize_astra_db(collection, embedding, astra_db_token, api_endpoint)

    print(f"Adding {len(split_documents)} documents to AstraDB...")
    vstore.add_documents(split_documents)

    # Retrieval
    my_prompt = prompt()
    retriever = as_retriever(vstore)
    model = chat_open_ai(model="gpt-3.5-turbo")

    print(f"Initializing model with prompt:\n{my_prompt}")
    chain = retrieval_chain(retriever, model, my_prompt)

    # Generation
    query_loop(chain)


# LLAMA_INDEX

from llama_index import (
    ServiceContext,
    LLMPredictor,
    OpenAIEmbedding,
    StorageContext,
    PromptHelper,
    VectorStoreIndex,
)
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import set_global_service_context
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import AstraDBVectorStore
from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.prompts import PromptTemplate


def llama_index_basic_rag_pipeline():
    # Initialize environment variables
    load_dotenv()
    astra_db_token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    documents = SimpleDirectoryReader(input_dir="./data").load_data()
    embed_model = OpenAIEmbedding()
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, text_splitter=text_splitter
    )

    astra_db = AstraDBVectorStore(
        collection_name="llama",
        token=astra_db_token,
        api_endpoint=api_endpoint,
        embedding_dimension=1536,
    )

    storage_context = StorageContext.from_defaults(vector_store=astra_db)

    vstore = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True,
    )

    # shakespeare!
    qa_prompt_str = """
    Given the context information and no prior knowledge,
    answer the query in the style of a Shakespeare passage.
    Keep the answer to 30 words or less.
    ---------------------
    {context_str}
    _____________________
    Query: {query_str}
    Answer:
    """
    qa_prompt = PromptTemplate(qa_prompt_str)

    # service context should already be set
    query_engine = vstore.as_query_engine(
        service_context=service_context,
        text_qa_template=qa_prompt,
    )

    # query_engine.update_prompts(
    #     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    # )

    while True:
        query = input("Query: ")
        response = query_engine.query(query)
        print(response)


llama_index_basic_rag_pipeline()
# rag_starter_app()
