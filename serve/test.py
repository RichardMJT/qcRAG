

from fastapi import FastAPI , WebSocket  # FastAPI 是一个为你的 API 提供了所有功能的 Python 类。
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, validator
import uvicorn
from httpservice import response

if __name__ == '__main__':
    app = FastAPI()  # 这个实例将是创建你所有 API 的主要交互对象。这个 app 同样在如下命令中被 uvicorn 所引用
    #解决跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],  # *：代表所有客户端
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    
    uvicorn.run("httpservice:app", host="127.0.0.1", port=8080, reload=True)