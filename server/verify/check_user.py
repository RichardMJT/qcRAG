import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath = os.path.dirname(temPath)    #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath)    

from server.db.session import with_async_session
from fastapi import HTTPException
from server.db.models.user_model import UserModel


@with_async_session
async def check_user(session, user_id:str, password: str):
    result = await session.get(UserModel, user_id)
    if not result:
        raise HTTPException(status_code=401, detail="user Id not found")
    return {"message":"User Id Exists"}