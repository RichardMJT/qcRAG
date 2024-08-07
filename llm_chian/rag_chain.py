import os
import sys
absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath01 = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath = os.path.dirname(temPath01)    #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath01)   
sys.path.append(temPath)   

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from model_to_llm import get_llm



## 简单rag执行 
def get_rag_chain(model, temperature, api_key):
    
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = get_llm(model, temperature, api_key)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain

