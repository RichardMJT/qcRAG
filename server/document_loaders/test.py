# from langchain_community.document_loaders import JSONLoader
# text_file = '/home/ezdx/文档/code/crag/server/document_loaders/json.json'
# loader =  JSONLoader(text_file)

# print(loader.load())

from langchain_community.document_loaders import JSONLoader
from pprint import pprint

def metadata_func(record: dict, metadata: dict) -> dict:
    # print(record)
    if "image_path" in record["metadata"]:
        # print(record)
        metadata["image_path"] = record["metadata"].get("image_path")
    # metadata["timestamp_ms"] = record.get("timestamp_ms")
    return metadata

loader = JSONLoader(
    file_path='/home/ezdx/文档/code/crag/knowledge_base/private/content/微检测WeD-1可视化手持核酸恒温荧光检测仪.json',
    jq_schema=".[]",
    content_key="text",
    text_content=False,
    metadata_func=metadata_func
)

data = loader.load()
pprint(data)
# from PIL import Image

# # 打开图片
# image = Image.open("/home/ezdx/文档/code/crag/knowledge_base/private/content/微检测WeD-1可视化手持核酸恒温荧光检测仪/pictures/figure-1-2.jpg")  # 替换为你的图片文件路径

# # 显示图片
# image.show()



# 打印结果
# for doc in documents:
#     print("Content:", doc.page_content)
#     print("Metadata:", doc.metadata)
