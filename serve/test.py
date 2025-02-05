

# from fastapi import FastAPI , WebSocket  # FastAPI 是一个为你的 API 提供了所有功能的 Python 类。
# from fastapi import Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field, ValidationError, validator
# import uvicorn
# from httpservice import response

# if __name__ == '__main__':
#     app = FastAPI()  # 这个实例将是创建你所有 API 的主要交互对象。这个 app 同样在如下命令中被 uvicorn 所引用
#     #解决跨域
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=['*'],  # *：代表所有客户端
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

    
#     uvicorn.run("httpservice:app", host="127.0.0.1", port=8080, reload=True)

import asyncio
import os
import sys
absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath01 = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath02 = os.path.dirname(temPath01)    #在往上返回一级，得到文件夹所在的路径
temPath03 = os.path.dirname(temPath02) #在往上返回一级，得到文件夹所在的路径
temPath04 = os.path.dirname(temPath03) #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath01)   
sys.path.append(temPath02)
sys.path.append(temPath03)
sys.path.append(temPath04)  


from database.get_vectordb import get_vectordb
from llm_chian.rag_chain import get_rag_chain
from llm_chian.retrieval_grader import get_retrieval_grader
from llm_chian.question_re_writer import get_question_rewriter
from tool.search import get_web_search_tool
from graph.crag import GraphPoint

file_path='../knowledge_db'
persist_path = '../vector_db/chroma'
api_key="sk-83f939a7ee424d588c176662a9636061"
embedding='bge'
model:str='qwen-max'
temperature:float=0.0
top_k:int=4
chat_history:list=[]
search_type="similarity"
search_kwargs={'k': 4}
# 创建向量数据库
# vectordb = get_vectordb(file_path, persist_path)
# 创建检索器
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService

# retriever = vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
retriever = FaissKBService("private")
rag_chain = get_rag_chain(model = model, temperature = temperature, api_key = api_key)
retrieval_grader = get_retrieval_grader(model = model, temperature = temperature, api_key = api_key)
question_rewriter = get_question_rewriter(model = model, temperature = temperature, api_key = api_key)
web_search_tool = get_web_search_tool()

# graph = GraphPoint(retriever, rag_chain, retrieval_grader, question_rewriter, web_search_tool)
# app = graph.bulid_graph()

# result = app.invoke({"question": "你好","chat_history": chat_history}) 

async def test_query():
    # search_ans = await retriever.search_docs(query="WeD-1")
    # print(search_ans)

    graph = GraphPoint(retriever, rag_chain, retrieval_grader, question_rewriter, web_search_tool)
    app = graph.bulid_graph()

    result =  await app.ainvoke({"question": "wed-1","chat_history": chat_history}) 
    print(result)

if __name__ == "__main__":
    asyncio.run(test_query())



