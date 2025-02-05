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

# import init_vs
# from knowledge_base.kb_service.faiss_kb_service import FaissKBService
# import asyncio
# from document_loaders.pdfloader import UnstructuredLightPipeline
# from server.knowledge_base.utils import KnowledgeFile
# from pathlib import Path

# 实例化 FaissKBService
# faiss_service = FaissKBService("private")
# async def a():
#     from server.knowledge_base.kb_service.base import KBServiceFactory
#     from server.db.repository.knowledge_base_repository import add_kb_to_db
#     # 先在Mysql中创建向量数据库的基本信息
#     await add_kb_to_db(kb_name="private",
#                            kb_info="个人/公司私有知识库数据",
#                            vs_type="faiss",
#                            embed_model="bge-large-zh-v1.5",
#                            user_id='admin')
# asyncio.run(a())


# async def b():
#     # faiss_service = FaissKBService("private")
#     processor = UnstructuredLightPipeline()
#     docs = await processor.run_pipeline('/home/ezdx/文档/code/fufan-chat-api/knowledge_base/private/content/invoice_1.pdf', ['unstructured'])
#     # kb_file = KnowledgeFile(Path('/home/ezdx/文档/code/fufan-chat-api/knowledge_base/private/content/invoice_1.pdf').name, "private")
#     # added_docs_info = await faiss_service.add_doc(kb_file, docs=docs)
#     print(docs)

# asyncio.run(b())

# kb_file = KnowledgeFile(Path('/home/ezdx/文档/code/fufan-chat-api/knowledge_base/private/content/invoice_1.pdf').name, "private")



# init_vs.process_and_add_document("/home/ezdx/文档/code/fufan-chat-api/knowledge_base/private/content/LangChain.pdf", faiss_service, 'admin')

# init_vs.private_main('admin')
# '''
# doc = [Document(metadata={'source': '/tmp/tmpocgeaa1m/file_data.txt'}, 
# page_content='Invoice no: 61356291\n\nDate of issue:\n\n09/06/2012\n\nSeller:\n\nClient:\n\nChapman, Kim and Green 64731 James Branch Smithmouth, NC 26872\n\nRodriguez-Stevens 2280 Angela Plain Hortonshire, MS 93248'),
# Document(metadata={'source': '/tmp/tmpocgeaa1m/file_data.txt'}, page_content='Tax Id: 949-84-9105 IBAN: GB50ACIE59715038217063\n\nTax Id: 939-98-8477\n\nITEMS'),]
# '''

import base64
from io import BytesIO
from PIL import Image

def read_base64_image(base64_string):
    # 解码Base64字符串为二进制数据
    image_data = base64.b64decode(base64_string)
    # 将二进制数据转换为图像对象
    image = Image.open(BytesIO(image_data))
    return image

# # 示例：读取Base64编码的图片
# base64_string = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="  # 示例Base64字符串
# image = read_base64_image(base64_string)
# image.show()  # 显示图片

import json

with open('/home/ezdx/文档/code/crag/knowledge_base/private/content/微检测WeD-1可视化手持核酸恒温荧光检测仪.json', 'r', encoding="utf-8") as file:
        data = json.load(file)
for entry in data:
      if 'image_base64' in entry['metadata']:
        base64_string = entry['metadata']['image_base64']
        image = read_base64_image(base64_string)
        image.show()  # 显示图片
        # break
