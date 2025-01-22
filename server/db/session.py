import os
import sys
absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath01 = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath02 = os.path.dirname(temPath01)    #在往上返回一级，得到文件夹所在的路径
temPath03 = os.path.dirname(temPath02) #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath01)   
sys.path.append(temPath02)
sys.path.append(temPath03)

from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
from server.db.base import AsyncSessionLocal
from functools import wraps

@asynccontextmanager
async def async_session_scope():
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise e
    finally:
        await session.close()

def with_async_session(f):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        async with async_session_scope() as session:
            return await f(session, *args, **kwargs)
    return wrapper

async def get_async_db():
    async with AsyncSessionLocal() as db:
        yield db
        



