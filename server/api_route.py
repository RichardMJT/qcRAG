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


import argparse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from server.chat.chat import chat
from server.db.repository.conversation_repository import create_conversation
from server.db.repository.user_repository import login_user




def create_app(run_mode: str = None):
    app = FastAPI(
        title="Crag Server",
    )
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载路由
    mount_app_routes(app)

    # 挂载 Vue 构建的前端静态文件夹
    # app.mount("/", StaticFiles(directory="static/dist"), name="static")
    return app

# @app.post("/")



def mount_app_routes(app: FastAPI):
    """
    这里定义通用领域问答对话的接口

    
    """

    # 用户登录
    app.post("/api/users/login",
             tags=["Users"],
             summary="用户登录",
             )(login_user)
    

    # 大模型对话接口
    app.post("/api/chat",
             tags=["Chat"],
             summary="大模型对话交互接口",
             )(chat)
    # 新建会话接口
    app.post("/api/conversations",
             tags=["Conversations"],
             summary="新建会话接口" 
             )(create_conversation)
    
# 

def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    
    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )



