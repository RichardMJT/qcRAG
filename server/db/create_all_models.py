import os
import sys
absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath01 = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath02 = os.path.dirname(temPath01)    #在往上返回一级，得到文件夹所在的路径
temPath03 = os.path.dirname(temPath02) #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath01)   
sys.path.append(temPath02)
sys.path.append(temPath03)

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from configs import SQLALCHEMY_DATABASE_URI

from server.db.base import Base
import server.db.models  # 确保模型被导入以便创建表
from sqlalchemy import Column, Integer, String, DateTime, JSON, func, ForeignKey
from server.db.base import async_engine, AsyncSessionLocal


from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, func, CHAR
from sqlalchemy.orm import relationship
from server.db.base import Base



async def create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(create_tables(async_engine))
