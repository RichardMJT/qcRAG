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

from document_loaders.interface import Pipeline
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from unstructured.staging.base import elements_to_json
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from pydantic import create_model
from typing import List
from rich.progress import Progress, SpinnerColumn, TextColumn
import tempfile
import json
import warnings
import yaml
import timeit
from rich import print
from typing import Any
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



class UnstructuredLightPipeline(Pipeline):
    '''
    本类的处理逻辑与 /home/ezdx/文档/code/fufan-chat-api/data/parse/unstructured_processor.py 类基本一致，
    知识将文件处理逻辑融入系统当中
    '''
    async def run_pipeline(self,
                            file_path: str,
                            options: List[str] = None,
                            debug: bool = True,
                            local: bool = True) -> Any:
        
        '''
        返回langchain切分后的对象
        '''
        print(f"\nRunning pipeline with unstructured_langchain \n")

        strategy = 'hi_res'
        model_name = 'yolox'

        extract_tables = False
        # Initialize options as an empty list if it is None
        options = options or []
        if "tables" in options:
            extract_tables = True

        # Extracts the elements from the PDF
        # 返回的elements是unstructured对象，接下来还要将其解析成json
        elements = self.invoke_pipeline_step(
            lambda: self.process_file(file_path, strategy, model_name),
            "Extracting elements from the document...",
            local
        )

        if debug:
            new_extension = 'json'  # You can change this to any extension you want
            new_file_path = self.change_file_extension(file_path, new_extension)
            # file_name_without_extension = os.path.splitext(file_path)[0]
            # new_file_path = file_name_without_extension + new_extension
            documents = self.invoke_pipeline_step(
                # load_text_data函数中，现将数据转为json在转成txt,在转为langchain能处理的对象
                lambda: self.load_text_data(elements, new_file_path, extract_tables),
                "Loading text data...",
                local
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, "file_data.json")

                documents = self.invoke_pipeline_step(
                    lambda: self.load_text_data(elements, temp_file_path, extract_tables),
                    "Loading text data...",
                    local
                )
    
        docs = self.invoke_pipeline_step(
            #将langchia能处理的对象型切分，如果修改切分方法要在这里修改
            lambda: self.split_text(documents, chunk_size=200, overlap=50),
            "Splitting text...",
            local
        )
        
        return docs

    def process_file(self, file_path, strategy, model_name):
        elements = None

        #提取文件名
        file_name_with_extension = os.path.basename(file_path)
        #去除文件名后缀
        file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
        #图像存储路径
        file_picture = f"/home/ezdx/文档/code/crag/knowledge_base/private/content/{file_name_without_extension}/pictures"
        if file_path.lower().endswith('.pdf'):
            # cishuunstructured的pdf解析器处理文件后获得的原始对象，还需要将其解析为json
            elements = partition_pdf(
                filename=file_path,
                strategy=strategy,
                # 表格数据以html形式返回
                infer_table_structure=True,
                model_name=model_name,
                # 提取图像
                extract_images_in_pdf=True,
                # 图像存放路径
                # ocr_languages = 'chi_sim'
                # extract_to_payload =True,
                image_output_dir_path = file_picture,
                # 定义语言
                languages=['chi_sim', 'eng']
            )


        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            elements = partition_image(
                filename=file_path,
                strategy=strategy,
                infer_table_structure=True,
                model_name=model_name
            )

        return elements

    def load_text_data(self, elements, file_path, extract_tables):
        '''
        该函数现将元素转为json,再将元素转为txt，名字容易引起误会，最终返回将langchainTextLoader包装后的txt对象
        这里返回的text不是原始文件，而是原始文件路径
        修改后不在返回text，而是返回json文件路径
        '''
        # 手动将元素保存到 JSON 文件中，确保使用 ensure_ascii=False
        # 在这里将unstructured_element转换为json
        with open(file_path, 'w', encoding='utf-8') as file:
            # 将 elements 列表中的每个元素 e 转换为字典格式
            # json.dump() 是 json 模块中的一个函数，用于将 Python 对象序列化为 JSON 格式，并将其写入到一个文件中
            json.dump([e.to_dict() for e in elements], file, ensure_ascii=False)

        
        
        # 功能与上一段代码相同，可以不用
        # elements_to_json(elements, filename=file_path)
        # 在这里将json转换为text,未来不在使用这种方法，而是直接对json进行切分处理
        # 这里返回的text不是原始文件，而是原始文件路径
        # 修改后不在返回text，而是返回json文件路径
        # 在这个函数中，使用多模态大模型对当前文件中包含的图片数据进行处理

        text_file = self.process_json_file(file_path, extract_tables)

        def metadata_func(record: dict, metadata: dict) -> dict:
            if "image_path" in record["metadata"]:
                metadata["image_path"] = record["metadata"].get("image_path")
            return metadata
        #加载为langchain能处理的对象
        # loader = TextLoader(text_file)
        loader =  JSONLoader(
            text_file,
            jq_schema=".[]",
            content_key="text",
            text_content=False,
            metadata_func=metadata_func
        )
        documents = loader.load()

        return documents

    # 在此处修改切分方案
    def split_text(self, text, chunk_size, overlap):
        '''
        # 之前的输入是txt文件，现在输入改为json文件，直接从json层面对数据进行处理
        # 在这里重写切分逻辑，切分好的将数据包装成langchain的Document对象，例如下面这样
        # Document(metadata={'source': '/tmp/tmpocgeaa1m/file_data.txt'}, page_content='Tax Id: 949-84-9105 IBAN: GB50ACIE59715038217063\n\nTax Id: 939-98-8477\n\nITEMS'),]

        将langchia能处理的对象进型切分，如果修改切分方法要在这里修改,返回切分后的对象
        '''
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        docs = text_splitter.split_documents(text)

        return docs

    def process_json_file(self, input_data, extract_tables):
        '''
        在这里将json转换为text,这个函数中不在进行这个操作
        当前这里返回的text不是原始文件，而是原始文件路径
        在这个函数中，使用多模态大模型对当前文件中包含的图片数据进行处理
        当前在这个函数中执行json转txt的完整过程，返回txt路径
        修改后不在返回text，而是返回json文件路径
        '''
        # Read the JSON file
        # json.load(file) 用于从文件对象中加载 JSON 数据，并将其解析为 Python 数据结构（如字典或列表）
        with open(input_data + '', 'r', encoding="utf-8") as file:
            data = json.load(file)



        #此处使用多模态大模型处理图片数据
        for entry in data:
            if entry["type"] == "Image":
                # 多模态大模型处理图片数据后返回的文本数据填充这个字段，并存入原始数据中
                entry["text"] = ""
                if  entry["text"] == "":
                    entry["text"] = "TMP"
        # 写入数据
        with open(input_data, 'w', encoding="utf-8") as file:
            json.dump(data, file, indent=4)# 将修改后的数据写入文件
            # file.write(str(data))  
        # Iterate over the JSON data and extract required table elements
        # extracted_elements = []
        # for entry in data:
        #     if entry["type"] == "Table":
        #         extracted_elements.append(entry["metadata"]["text_as_html"])
        #     elif entry["type"] == "Title" and extract_tables is False:
        #         extracted_elements.append(entry["text"])
        #     elif entry["type"] == "NarrativeText" and extract_tables is False:
        #         extracted_elements.append(entry["text"])
        #     elif entry["type"] == "UncategorizedText" and extract_tables is False:
        #         extracted_elements.append(entry["text"])

        # # # Write the extracted elements to the output file
        # new_extension = 'txt'  # You can change this to any extension you want
        # new_file_path = self.change_file_extension(input_data, new_extension)
        # # # new_file_path = os.path.splitext(input_data)[0] + new_extension
        # with open(new_file_path, 'w') as output_file:
        #     for element in extracted_elements:
        #         output_file.write(element + "\n\n")  # Adding two newlines for separation
        
        # return new_file_path
        return input_data

    def change_file_extension(self, file_path, new_extension):
        # Check if the new extension starts with a dot and add one if not
        if not new_extension.startswith('.'):
            new_extension = '.' + new_extension

        # # Split the file path into two parts: the base (everything before the last dot) and the extension
        # # If there's no dot in the filename, it'll just return the original filename without an extension
        base = file_path.rsplit('.', 1)[0]
        # base = '/home/ezdx/文档/code/crag/knowledge_base/private'

        # # Concatenate the base with the new extension
        new_file_path = base + new_extension

        return new_file_path

    def beautify_json(self, result):
        try:
            # Convert and pretty print
            data = json.loads(str(result))
            data = json.dumps(data, indent=4)
            return data
        except (json.decoder.JSONDecodeError, TypeError):
            print("The response is not in JSON format:\n")
            print(result)

        return {}

    def invoke_pipeline_step(self, task_call, task_description, local):
        if local:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=False,
            ) as progress:
                progress.add_task(description=task_description, total=None)
                ret = task_call()
        else:
            print(task_description)
            ret = task_call()

        return ret


