import streamlit as st
import torch 
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from helper import get_embeddings

#need to do from first

device_type="cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = ""
MODEL_BASENAME = ""
EMBEDDING_MODEL_NAME = ""

#embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})





#only single page we have to make this for multiple html  pages 
def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    emb=get_embeddings(EMBEDDING_MODEL_NAME,device_type)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, emb)

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain


def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

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