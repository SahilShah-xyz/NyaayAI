from typing import List

import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
)
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma


@cl.cache
def load_db():
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    model_kwargs = {"device": "cuda:0", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": False}
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    docsearch = Chroma(
        collection_name="rag-chroma",
        persist_directory="/workspace/persist_dir",
        # persist_directory="/content/drive/MyDrive/Experiments/LLMChatbot/persist_dir",
        embedding_function=model,
    )
    return docsearch


@cl.on_chat_start
async def on_chat_start():
    docsearch = load_db()

    # repo_id = "google/gemma-2b"
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
    )
    # llm = HuggingFaceEndpoint(
    #     endpoint_url="https://api-inference.huggingface.co/models/Sahi19/Gemma2bLegalChatbot",
    #     task="text-generation",
    # )

    message_history = ChatMessageHistory()
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
    )

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke(message.content, callbacks=[cb])
    print(res)
    answer = res["answer"]
    # source_documents = res["source_documents"]

    # text_elements = []

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    #     if source_names:
    #         answer += f"\nSources: {', '.join(source_names)}"
    #     else:
    #         answer += "\nNo sources found"

    # await cl.Message(content=answer, elements=text_elements).send()
    await cl.Message(content=answer).send()
