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

# from server.db.models.user_model import UserModel
# from server.db.models.conversation_model import ConversationModel
# from server.db.models.message_model import MessageModel
# from server.db.models.knowledge_base_model import KnowledgeBaseModel
# from server.db.models.knowledge_file_model import KnowledgeFileModel
# from server.db.models.knowledge_file_model import FileDocModel


from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, func, CHAR
from sqlalchemy.orm import relationship
from server.db.base import Base


# class UserModel(Base):
#     __tablename__ = 'user'
#     id = Column(CHAR(36), primary_key=True, comment='用户ID')
#     username = Column(String(255), unique=True, comment='用户名')
#     password_hash = Column(String(255), comment='密码的哈希值')
#     # 可以添加更多用户相关的字段，如邮箱、电话等

#     conversations = relationship('ConversationModel', back_populates='user')
#     knowledge_bases = relationship('KnowledgeBaseModel', back_populates='user', cascade='all, delete-orphan')  # 更新关系

#     def __repr__(self):
#         return f"<User(id='{self.id}', username='{self.username}')>"

async def create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(create_tables(async_engine))
