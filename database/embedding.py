from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
import sys
absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath = os.path.dirname(temPath)    #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath)    
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'


## 从HuggingFace获取使用向量化模型
def get_embedding(embedding: str):
    if embedding == 'bge':
        model_name = 'BAAI/bge-large-zh-v1.5'
        model_kwargs = {'device': 'cuda'}  # 需要安装GPU版本的torch ，如果没有，这里cuda改为cpu
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embeddings
    else:
        raise ValueError(f"embedding {embedding} not support ")
