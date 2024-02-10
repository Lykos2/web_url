from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from helper import get_embeddings
import torch 


device_type="cuda" if torch.cuda.is_available() else "cpu"


EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)
embeddings=get_embeddings(EMBEDDING_MODEL_NAME,device_type)
query = "What is the website is about"

DB2 = Chroma(
        persist_directory='./DB',
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
docs = DB2.similarity_search(query)
print(DB2._collection.count())