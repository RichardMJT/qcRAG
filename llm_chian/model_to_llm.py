import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from dotenv import find_dotenv, load_dotenv

# load_dotenv(find_dotenv())

def get_llm(model:str=None, temperature:float=0.0, api_key:str=None):
    llm = None
    if model == 'qwen-max':
        llm = ChatTongyi(model=model, temperature = temperature, api_key= api_key)
    if model == 'llama3.1':
        llm = ChatOllama(model=model, temperature = temperature)
    if model == 'qwen2':
        llm = ChatOllama(model=model,temperature = temperature)
    return llm