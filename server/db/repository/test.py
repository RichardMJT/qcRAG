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
# print(temPath04)

from pydantic import BaseModel, Field
from fastapi import HTTPException, Depends, Body
from sqlalchemy.exc import IntegrityError
from server.db.session import get_async_db
from server.db.models.user_model import UserModel
from passlib.hash import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
import uuid
from fastapi import Response
from fastapi.responses import JSONResponse
from typing import List
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from server.db.session import with_async_session
from fastapi import HTTPException
from server.db.models.user_model import UserModel

class UserLoginRequest(BaseModel):
    username: str = Field(..., example="user123")
    password: str = Field(..., example="password123")

# async def login_user(
#         request: UserLoginRequest = Body(...),
#         session: AsyncSession = Depends(get_async_db)
# ):
#     print(request)
#     print(session)
#     # 使用 username 来查询用户
#     user = await session.execute(select(UserModel).where(UserModel.username == request.username))
#     user = user.scalar_one_or_none()

#     if user and bcrypt.verify(request.password, user.password_hash):

#         return JSONResponse(
#             status_code=200,
#             content={
#                 "status": 200,
#                 "id": user.id,
#                 "username": user.username,
#                 "message": "Login successful"
#             }
#         )
#     else:
#         return {"status": 401, "message": "用户名或密码错误。"}
#         # raise HTTPException(status_code=401, detail="Invalid username or password")
# import asyncio

# async def main():
#     tmp = UserLoginRequest(username="newuser456", password="newsecurepassword456")
#     await login_user(tmp)

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse
import bcrypt
from server.db.base import AsyncSessionLocal

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker

from configs import SQLALCHEMY_DATABASE_URI
import json

#解决跨域


async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URI,
    echo=True,
)


# AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

# FastAPI应用实例
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # *：代表所有客户端
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



Session = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
async def get_db_session():
    
    session = Session()
    try:
        yield session
    finally:
        session.close()



# 登录路由
@app.get("/login")
async def login(
    # request: UserLoginRequest = Body(...),
    username:str,
    password:str,
    session: AsyncSession = Depends(get_db_session)
):
    request = UserLoginRequest(username=username, password=password)
    user = await session.execute(select(UserModel).where(UserModel.username == username))
    # print(user)
    # print('1111111')
    # return ''
    return await login_user(request, session)

# login_user函数定义（与之前给出的相同）
async def login_user(
       request, 
       session
):
    # session = await get_async_db()
    # session = session()
    # 使用 username 来查询用户
    user = await session.execute(select(UserModel).where(UserModel.username == request.username))
    user = user.scalar_one_or_none()
    print()
    # if user and bcrypt.checkpw(request.password, user.password_hash):

    if user :
        return JSONResponse(
            status_code=200,
            content={
                "status": 200,
                "id": user.id,
                "username": user.username,
                "message": "Login successful"
            }
        )
        # return user.password_hash
    else:
        return {"status": 401, "message": "用户名或密码错误。"}
        # raise HTTPException(status_code=401, detail="Invalid username or password")


# 测试函数
async def test_login():
    # 创建一个模拟的请求对象
    request = UserLoginRequest(username='admin', password='admin')
    # response = await login(request)
    
    
    # 调用login_user函数
    try:
        response = await login(request)
        print(response)
        print('11111')
    except HTTPException as e:
        print(f"Error: {e.detail}")

# # 运行测试函数
# if __name__ == "__main__":
#     asyncio.run(test_login())


if __name__ == '__main__': 
    import uvicorn
    uvicorn.run("test:app", host="127.0.0.1", port=8080, reload=True)