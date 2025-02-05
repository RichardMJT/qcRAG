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


from langgraph.prebuilt import create_react_agent
from llm_chian.model_to_llm import get_llm
from configs.model_config import API_KEY, TEMPERATURE
from tool.detected_data_search import query_sales
from langchain_core.messages import  AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage


def get_db_agent(llm):
    # llm = get_llm(model = 'qwen-max', temperature = TEMPERATURE, api_key = API_KEY)

    db_agent = create_react_agent(
        llm, 
        tools=[query_sales], 
        state_modifier="您在执行数据库操作时，应该为代码生成器提供准确的数据"
        # debug = True
    )

    return db_agent

async def main():
    llm = get_llm(model = 'qwen-max', temperature = TEMPERATURE, api_key = API_KEY)
    agent= get_db_agent(llm)
    messages = {"messages":[SystemMessage(content="用户ID:admin"), HumanMessage(content="有多少次检测结果为阳性？")]}
    
    res = await agent.ainvoke(messages)
    print(res["messages"][-2])
    pass

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())