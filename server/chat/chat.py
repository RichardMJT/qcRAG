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

import asyncio
from fastapi import Body, HTTPException
from typing import List, Union, Optional
from sse_starlette.sse import EventSourceResponse
from configs.model_config import LLM_MODELS, TEMPERATURE, MAX_TOKENS, STREAM


from typing import AsyncIterable
import json
from langchain.callbacks import AsyncIteratorCallbackHandler

from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
import uuid
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from server.db.session import get_async_db
from server.db.repository.message_repository import add_message_to_db
from langchain.prompts import PromptTemplate
from serve.run_gradio import model_center
from server.db.repository import update_message

from pydantic import BaseModel, Field


class Chat(BaseModel):
    user_id: str = Field(..., description="用户ID")
    query: str = Field(..., description="用户输入")
    conversation_id: str = Field("", description="对话框ID")
    model_name: str = Field("", description="LLM 模型名称。")
    prompt_name: str = Field("general_chat",
                                       description="使用的prompt模板名称(在configs/prompt_config.py中配置)")

    


async def chat(chat:Chat):

    # 创建graph


    # # 构造一个新的Message_ID记录
    message_id = await add_message_to_db(query=chat.query,
                                             conversation_id=chat.conversation_id,
                                             prompt_name=chat.prompt_name
                                             )
    #返回答案 
    answer = await model_center.get_answer(chat.query)
    #存入数据库
    await update_message(message_id=message_id, response=answer)
    # pass
    return {"answer":answer}