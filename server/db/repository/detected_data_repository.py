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


from server.db.session import with_async_session, async_session_scope
from typing import Dict, List
import uuid
from server.db.models.detected_data_model import DetectedDataModel
from sqlalchemy.future import select


# 根据id搜索数据库
@with_async_session
async def get_message_by_id(session, user_id) -> DetectedDataModel:
    """
    Asynchronously query a chat record by ID
    """
    result = await session.execute(select(DetectedDataModel).filter_by(user_id=user_id))
    return result.scalars().all()

async def main(user_id):
    res = await get_message_by_id(user_id)
    print(res[0].id)


if __name__ =='__main__':
    import asyncio
    asyncio.run(main("admin"))
    pass