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

import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import  AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage
from llm_chian.model_to_llm import get_llm
from configs.model_config import API_KEY, TEMPERATURE
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,   # 以 JSON 形式返回函数调用的参数
    JsonOutputToolsParser,      # 以 JSON 形式返回函数调用中特定键的值
    PydanticToolsParser,        # 将函数调用的参数作为 Pydantic 模型返回
)
from langgraph.prebuilt import create_react_agent
from configs.kb_config import SQLALCHEMY_DATABASE_URI
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from server.db.models.detected_data_model import DetectedDataModel
from server.db.session import with_async_session
from langchain_core.tools import BaseTool



class QuerySalesSchema(BaseModel):
    '''
        用户ID
    '''
    user_id: str = Field(description="用户ID")

@tool(args_schema=QuerySalesSchema)
@with_async_session
async def query_sales(session, user_id):
    '''
        查询用户与检测有关操作信息的函数
        :user_id字符串类型，用于表示发出查询请求的用户ID
        返回值是json格式的字符串，包含检测相关信息
    '''
    res = await session.execute(select(DetectedDataModel).filter_by(user_id=user_id))
    res = res.scalars().all()

    if not res:
        return json.dumps({"data": []})
    # print(user_id)
    else:
        data = [
            {
                "id": msg.id,
                "user_id":msg.user_id,
                "detected_name":msg.detected_name,
                "create_time":msg.create_time.isoformat(),
                "operating_data":msg.operating_data,
                "result":msg.result,
                "location":msg.location
            } for msg in res
        ]
    return json.dumps({"data": data})

async def main():
    llm = get_llm(model = 'qwen-max', temperature = TEMPERATURE, api_key = API_KEY)

    db_agent = create_react_agent(
        llm, 
        tools=[query_sales], 
        state_modifier="",
        debug = True
    )

    # # llm_with_tools = llm.bind_tools([query_sales])

    messages = {"messages":[SystemMessage(content="用户ID:admin"), HumanMessage(content="查找所有检测结果为阳性的记录")]}
    res = await db_agent.ainvoke(messages)
    # # res = llm_with_tools.invoke(messages)
    # # cian = llm_with_tools | JsonOutputKeyToolsParser(key_name='query_sales', first_tool_only=True) | query_sales
    # # res = cian.invoke(messages)
    
    # res = await query_sales.ainvoke("admin")
    print(res)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
    # asyncio.run()
    # query_sales("admin")