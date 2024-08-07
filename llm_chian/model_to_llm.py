import os
from langchain_community.chat_models.tongyi import ChatTongyi

# from dotenv import find_dotenv, load_dotenv

# load_dotenv(find_dotenv())

def get_llm(model:str=None, temperature:float=0.0, api_key:str=None):
    llm = ChatTongyi(model=model, temperature = temperature, api_key= api_key)
    return llm