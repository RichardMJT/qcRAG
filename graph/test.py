

# import os
# import sys
# absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
# absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
# temPath01 = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
# temPath02 = os.path.dirname(temPath01)    #在往上返回一级，得到文件夹所在的路径
# temPath03 = os.path.dirname(temPath02) #在往上返回一级，得到文件夹所在的路径
# sys.path.append(temPath01)   
# sys.path.append(temPath02)
# sys.path.append(temPath03)

# from typing_extensions import TypedDict
# from typing import Literal
# from langchain_openai import ChatOpenAI
# from llm_chian.model_to_llm import get_llm

# members = ["chat", "ve_RAG"]
# options = members + ["FINISH"]




# class Router(TypedDict):
#     """Worker to route to next. If no workers needed, route to FINISH"""
#     # 节点名称
#     next: Literal["chat", "ve_RAG", "FINISH"]

# system_prompt = (
#             "You are a supervisor tasked with managing a conversation between the"
#             f" following workers: {members}.\n\n"
#             "Each worker has a specific role:\n"
#             "- chat: Responds directly to user inputs using natural language.\n"
#             "- ve_RAG: vec_kg: Stores market and company information, constructed on a traditional semantic retrieval knowledge base, excels at answering detailed and fine-grained questions.\n"
#             "Given the following user request, respond with the worker to act next."
#             " Each worker will perform a task and respond with their results and status."
#             " When finished, respond with FINISH."
#         )


# llm = get_llm(model = "llama3.3-70b-instruct")

# messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "你好"}] 

# # response = await llm.with_structured_output(Router).ainvoke(messages)
# response = llm.with_structured_output(Router).invoke(messages)
# print(response)



# 导入检查点
from typing import Annotated
from langgraph.checkpoint.memory import MemorySaver
import os
from langgraph.graph.message import add_messages

import asyncio
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
# from llm_chian.model_to_llm import get_llm
from langchain_openai import ChatOpenAI
from langchain_core.messages import  AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage


llm = ChatOpenAI(
                api_key='sk-83f939a7ee424d588c176662a9636061',
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model='qwen-max',
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

async def call_model(state: State):
    messages = state["messages"][-1]
    tmp = [HumanMessage(content=messages.content)]
    response = await llm.ainvoke(state["messages"])
    response = [AIMessage(content=response.content)]
    return {"messages": response, "next": "111"}

async def translate_message(state: State):
    system_prompt = """
    Please translate the received text in any language into English as output
    """
    messages = state['messages'][-1]
    messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=messages.content)]
    response = await llm.ainvoke(messages)
    response = [AIMessage(content=response.content)]
    return {"messages": response, "next": "111"}

builder = StateGraph(State)

builder.add_node("call_model", call_model)
builder.add_node("translate_message", translate_message)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "translate_message")
builder.add_edge("translate_message", END)

async def main():
    # memory = MemorySaver()
    # graph_with_memory = builder.compile(checkpointer=memory)   # 在编译图的时候添加检查点
    memory = MemorySaver()
        # 编译图
    graph_with_memory = builder.compile(checkpointer = memory)

    config = {"configurable": {"thread_id": "1"}}

    async for chunk in graph_with_memory.astream(input={"messages": ["你好，我叫木羽"]}, config=config, stream_mode="values"):
        print(chunk)
        print()

    async for chunk in graph_with_memory.astream(input={"messages": ["请问我叫什么？"]}, config=config, stream_mode="values"):
        print(chunk)
        print()

if __name__ == "__main__":
    asyncio.run(main())