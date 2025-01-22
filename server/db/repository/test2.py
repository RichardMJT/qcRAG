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

import uvicorn
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker,Session
from configs import SQLALCHEMY_DATABASE_URI
from sqlalchemy.future import select
from server.db.models.user_model import UserModel

engine = create_engine(SQLALCHEMY_DATABASE_URI)
Sessions = sessionmaker(bind=engine)

app = FastAPI()


async def get_db_session():
    
    session = Sessions()
    try:
        yield session
    finally:
        session.close()
@app.post("/a")
async def get_students(session:  Session = Depends(get_db_session)):
    user = session.execute(select(UserModel).where(UserModel.username == 'admin'))
    print(user)
    return "111"

if __name__ == '__main__': 
    
    uvicorn.run("test2:app", host="127.0.0.1", port=8080, reload=True)