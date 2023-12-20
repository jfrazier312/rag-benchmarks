from typing import Optional

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import PromptTemplate

from rag_benchmarks.retrieval.prompts import CHAT_PROMPT_SUFFIX, CONVERSATION_PROMPT_SUFFIX


def as_retriever(vstore: VectorStore, k: Optional[int] = 4) -> VectorStoreRetriever:
    """
    Convert a VectorStore into a VectorStoreRetriever.

    Args:
        vstore (VectorStore): The VectorStore to be converted into a retriever.
        k (Optional[int]): Amount of documents to return
            Default is 4 if not specified.

    Returns:
        VectorStoreRetriever: A retriever instance.
    """
    return vstore.as_retriever(search_kwargs={"k": k})


def basic_chat(retriever: VectorStoreRetriever, llm: BaseChatModel, prompt: str):
    chat_prompt = prompt + CHAT_PROMPT_SUFFIX
    chat_prompt_template = ChatPromptTemplate.from_template(chat_prompt)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


def basic_chat_with_memory(
    retriever: VectorStoreRetriever, llm: BaseChatModel, prompt: str
):
    conversation_prompt = prompt + CONVERSATION_PROMPT_SUFFIX
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=conversation_prompt,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": prompt_template},
        # verbose=True,
        # return_source_documents=True,
    )

    return chain
