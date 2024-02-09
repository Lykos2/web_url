import torch
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from helper import get_embeddings
import streamlit as st 
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from helper import load_model 
from langchain.chains import RetrievalQA    









device_type="cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory



if "EMBEDDINGS" not in st.session_state:
    EMBEDDINGS = get_embeddings(EMBEDDING_MODEL_NAME)
    st.session_state.EMBEDDINGS = EMBEDDINGS



if "DB" not in st.session_state:
    DB = Chroma(
        persist_directory='./DB',
        embedding_function=st.session_state.EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )
    st.session_state.DB = DB

if "RETRIEVER" not in st.session_state:
    RETRIEVER = DB.as_retriever()
    st.session_state.RETRIEVER = RETRIEVER


if "LLM" not in st.session_state:
    LLM = load_model(device_type=device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
    st.session_state["LLM"] = LLM


if "QA" not in st.session_state:
    prompt, memory = model_memory()

    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    st.session_state["QA"] = QA




st.set_page_config(
    page_title="chat_with_website"
)

st.title("chat with website")

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = st.session_state["QA"](user_query)
    answer, docs = response["result"], response["source_documents"]
    st.write(answer)
    with st.expander("Document Similarity Search"):
    # Find the relevant pages
        search = st.session_state.DB.similarity_search_with_score(user_query)
        # Write out the first
        for i, doc in enumerate(search):
            # print(doc)
            st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
            st.write(doc[0].page_content)
            st.write("--------------------------------")







