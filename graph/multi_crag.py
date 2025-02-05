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
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

import asyncio
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import  AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from llm_chian.model_to_llm import get_llm

from configs.model_config import TEMPERATURE, API_KEY

#引入RAG结点
from graph.crag import GraphPoint

from database.get_vectordb import get_vectordb
from llm_chian.rag_chain import get_rag_chain
from llm_chian.retrieval_grader import get_retrieval_grader
from llm_chian.question_re_writer import get_question_rewriter
from tool.search import get_web_search_tool

# 创建检索器
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService

from sql_graph import get_db_agent


class AgentState(MessagesState):
    '''
        从MessagesState集成有messages: Annotated[list[AnyMessage], add_messages]
        所以这个类实际上包含两个字段
        messages: Annotated[list[AnyMessage], add_messages]
        next: str
    '''
    next: str

# class AgentState(TypedDict):
#     messages: Annotated[list, add_messages]
#     next: str


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    # 节点名称
    next: Literal["chat", "ve_RAG", "db_agent", "FINISH"]

class MultiRAG():
    def __init__(self, user_id, model:str='qwen-max' ):
        self.llm = get_llm(model = model, temperature = TEMPERATURE, api_key = API_KEY)
        self.members = ["chat", "ve_RAG","db_agent"]
        self.options = self.members + ["FINISH"]
        self.user_id = user_id
        
        
        retriever = FaissKBService("private")
        rag_chain = get_rag_chain(model = model, temperature = TEMPERATURE, api_key = API_KEY)
        retrieval_grader = get_retrieval_grader(model = model, temperature = TEMPERATURE, api_key = API_KEY)
        question_rewriter = get_question_rewriter(model = model, temperature = TEMPERATURE, api_key = API_KEY)
        web_search_tool = get_web_search_tool() 


        # 创veRAG智能体
        self.ve_RAG_graph = GraphPoint(retriever=retriever, rag_chain=rag_chain, retrieval_grader=retrieval_grader, question_rewriter=question_rewriter, web_search_tool=web_search_tool).bulid_graph()
        self.sql_db_agent = get_db_agent(self.llm)

    
    async def supervisor(self, state: AgentState):
        # system_prompt = (
        #     "You are a supervisor tasked with managing a conversation between the"
        #     f" following workers: {self.members}.\n\n"
        #     "Each worker has a specific role:\n"
        #     "- chat: Responds directly to user inputs using natural language.\n"
        #     "- ve_RAG :Stores market and company information, constructed on a traditional semantic retrieval knowledge base, excels at answering detailed and fine-grained questions.\n"
        #     "Given the following user request, respond with the worker to act next."
        #     " Each worker will perform a task and respond with their results and status."
        #     " When finished, respond with FINISH."
        # )

        system_prompt = (
            "你是负责管理以下智能体的主管智能体："
            f"{self.members}。\n\n"
            "每个智能体都有特定的角色：\n"
            "- chat：使用自然语言直接回应用户输入。"
            #"- search:如果智能体chatwu无法准确的回答阿用户问题，或者在搜索知识库后如果没有得到满足条件的信息，使用这个智能体。"
            "- ve_RAG：存储杭州逸检科技有限公司信息，包括公司简介、WeD系列仪分子诊断仪器等逸检科技出售的有关产品详细信息和使用说明，基于传统语义检索知识库构建，擅长回答详细且细致的问题。"
            "- db_agent：能够检索/查询关系型数据库，获取与历史检测相关的记录，并根据记录回答问题。\n"
            "根据以下用户请求，指定下一位执行任务的智能体。\n"
            "每个智能体将执行任务并返回结果和状态。\n"
            "任务完成后，回复“FINISH\n"
        )

        messages = [{"role": "system", "content": system_prompt},] + state["messages"]

        response = await self.llm.with_structured_output(Router).ainvoke(messages)

        next_ = response["next"]
        
        if next_ == "FINISH":
            next_ = END
        
        return { "next": next_}


    async def chat(self, state: AgentState):
        messages = state["messages"]
        model_response = await self.llm.ainvoke(messages)
        final_response = [AIMessage(content=model_response.content, name="chat")]
        return {"messages": final_response}


    async def ve_RAG(self, state: AgentState):
        print('============state["messages"][-1]===========')
        print(state["messages"][-1].content)
        print('============state["messages"][-1]===========')

        result = await self.ve_RAG_graph.ainvoke({"question": state["messages"][-1].content})
        # for chunck in await self.ve_RAG_graph.ainvoke({"question": state["messages"][-1].content}):
        #     result.append(chunck)
        # result = await self.ve_RAG_graph.ainvo  ke({"question": state["messages"][-1].content})
        # return {"messages": [AIMessage(content=result.generation, name="ve_RAG")]}
        # print("=================result=======================")
        # print(type(result))
        # print(result['generation'])
        # print("================result=====================")
        
        return {"messages": [AIMessage(content=result['generation'], name="ve_RAG")]}
    
    async def db_agent(self, state:AgentState):
        # db_agent = get_db_agent(self.llm)
        content = f"当前与程序对话的用户ID为:{self.user_id}"
        messages = {"messages":[SystemMessage(content=content), HumanMessage(content=state["messages"][-1].content)]}
        # print(messages)
        result = await self.sql_db_agent.ainvoke(messages)
        return {"messages": [AIMessage(content=result["messages"][-1].content, name="db_agent")]}

    def bulid_graph(self):

        
        builder = StateGraph(AgentState)
        # 添加开始和节点
        builder.add_edge(START, "supervisor")
        builder.add_node("supervisor", self.supervisor)
        builder.add_node("chat", self.chat)
        builder.add_node("ve_RAG", self.ve_RAG)
        builder.add_node("db_agent", self.db_agent)


        for member in self.members:
            # 我们希望我们的工人在完成工作后总是向主管智能体“汇报”
            builder.add_edge(member, "supervisor")
        
        # 在图状态中填充next字段，路由到具体的某个节点或者结束图的运行，指定如何执行接下来的任务
        builder.add_conditional_edges("supervisor", lambda state: state["next"])

        # 添加开始和节点
        # builder.add_edge(START, "supervisor")

        # builder.add_edge(START, "chat")
        # 在创建任何 LangGraph 图时，通过在编译图时添加MemorySaver来将其设置为保留其State状态中的数据
        memory = MemorySaver()
        # 编译图
        graph = builder.compile(checkpointer = memory)
        return graph



async def main():

    tmp = MultiRAG(user_id = "admin")
    graph = tmp.bulid_graph()
    # from IPython.display import Image, display
    # from PIL import Image
    # img = Image.open(graph.get_graph(xray=True))
    # img.show()
    message = [SystemMessage(content='你是杭州逸检科技有限公司开发的智能助手，可以帮助与你对话的用户解决与逸检科技有关的各种问题。'), HumanMessage(content="查找所有检测结果为阳性的记录")]

    n_msg = AgentState(messages=message) 
    
    print('----------------')
    config = {"configurable": {"thread_id": "2"}}
    async for chunk in graph.astream(input=n_msg, config=config, stream_mode="values"):
        # pass
        print(chunk)
        print()
    # message = [HumanMessage(content="介绍一下WeD-mini")]
    # n_msg = AgentState(messages=message)
    # async for chunk in graph.astream(input=n_msg, config=config, stream_mode="values"):
    #     print(chunk)
    #     print()
   


if __name__ == "__main__":
    asyncio.run(main())




