import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# from dotenv import find_dotenv, load_dotenv

# load_dotenv(find_dotenv())

def get_llm(model, temperature, api_key):
    llm = None
    if model == 'qwen-max' or model == 'qwen-plus' :
        llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model=model,
            )
    elif model == 'llama3.1':
        llm = ChatOllama(model=model, temperature = temperature)
    elif model == 'qwen2':
        llm = ChatOllama(model=model,temperature = temperature)
    return llm

if __name__ == "__main__":
    get_llm('qwen-max')