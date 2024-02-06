import os
import io 
import torch 
from PyPDF2 import PdfReader
import logging 
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
from loading_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)
CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

#### If you get a "not enough space in the buffer" error, you should reduce the values below, start with half of the original values and keep halving the value until the error stops appearing

N_GPU_LAYERS = 100  # Llama-2-70B has 83 layers
N_BATCH = 512


import streamlit as st


device="cuda" if torch.cuda.is_available() else "cpu"

def get_file_list(folder_path):
    files_in_folder=os.listdir(folder_path)
    names = [filename for filename in files_in_folder if os.path.isfile(os.path.join(folder_path, filename))]
    names.append("other")
    return names

def read_file_to_list(file_path):
    try:
        # Initialize an empty list to store the lines from the file
        lines = []

        # Open the file in read mode ('r')
        with open(file_path, 'r') as file:
            # Read each line and append it to the 'lines' list
            for line in file:
                lines.append(line.strip())  # Use .strip() to remove newline characters

        # Return the list containing the lines from the file
        lines.append("other")
        return lines
    except FileNotFoundError:
        # Handle the case where the file does not exist
        return []
    except Exception as e:
        # Handle other exceptions, e.g., permission errors
        print(f"An error occurred: {e}")
        return []


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):

    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

def process_pdf(pdf_reader):
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()



def get_embeddings(EMBEDDING_MODEL_NAME,device_type="cuda"):
    if "instructor" in EMBEDDING_MODEL_NAME:
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            embed_instruction="Represent the document for retrieval:",
            query_instruction="Represent the question for retrieving supporting documents:",
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            query_instruction="Represent this sentence for searching relevant passages:",
        )

    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
        )