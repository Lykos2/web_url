import torch 
from langchain_community.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncHtmlLoader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from helper import get_embeddings




EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"


CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

device_type="cuda" if torch.cuda.is_available() else "cpu"


def get_all_links(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the page: {url}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    # Finding all 'a' tags which typically contain href attribute for links
    links = [
        urljoin(url, a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"]
    ]

    return links




def main(url):
    
    all_links=get_all_links(url)

    url_pattern = re.compile(r'https?://\S+')
    links = [url for url in all_links if url_pattern.match(url)]
    links=links[0:100]

    loader = AsyncHtmlLoader(links)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(docs_transformed)
    embeddings=get_embeddings(EMBEDDING_MODEL_NAME,device_type)
    db = Chroma.from_documents(
        document_chunks,
        embeddings,
        persist_directory="./DB",
        client_settings=CHROMA_SETTINGS,
    )
    
    # create a vectorstore from the chunks
if __name__ == "__main__":
    url = "https://www.thomascook.in/"
    main(url)