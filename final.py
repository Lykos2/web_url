import streamlit as st
import torch 
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma




device_type="cuda" if torch.cuda.is_available() else "cpu"


#embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})





#only single page we have to make this for multiple html  pages 

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(doc)
    vector_store=Chroma.from_documents(document_chunks,)
    return document_chunks




def get_response(user_input):
    return "I dont know"

st.set_page_config(
    page_title="chat_with_website"
)

st.title("chat with website")


if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
    AIMessage(content="Hello how can i help you")]


with st.sidebar:
    st.header("settings")
    website_url=st.text_input("Website URL")


if website_url is None or website_url == "":
    st.info("Please enter the url")
else:
    docs=get_vectorstore_from_url(website_url)
    with st.sidebar:
        st.write(docs)


    user_query=st.chat_input("Enter Your Message")

    if user_query is not None and user_query!="":
        response=get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("BOT"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)