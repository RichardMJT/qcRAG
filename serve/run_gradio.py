import os
import sys
import gradio as gr

absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath01 = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath = os.path.dirname(temPath01)    #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath01)   
sys.path.append(temPath)   

from database.get_vectordb import get_vectordb
from llm_chian.rag_chain import get_rag_chain
from llm_chian.retrieval_grader import get_retrieval_grader
from llm_chian.question_re_writer import get_question_rewriter
from tool.search import get_web_search_tool
from graph.crag import GraphPoint
import re

file_path='../knowledge_db'
persist_path = '../vector_db/chroma'
api_key=''
embedding='bge'

class model_center():
    def __init__(self,model:str='qwen-max', temperature:float=0.0, top_k:int=4, chat_history:list=[], search_type="similarity", search_kwargs={'k': 4},):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history

        # 创建向量数据库
        vectordb = get_vectordb(file_path, persist_path)
        # 创建检索器
        self.retriever = vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        self.rag_chain = get_rag_chain(model = model, temperature = temperature, api_key = api_key)
        self.retrieval_grader = get_retrieval_grader(model = model, temperature = temperature, api_key = api_key)
        self.question_rewriter = get_question_rewriter(model = model, temperature = temperature, api_key = api_key)
        self.web_search_tool = get_web_search_tool()

    def get_graph(self):
        graph = GraphPoint(self.retriever, self.rag_chain, self.retrieval_grader, self.question_rewriter, self.web_search_tool)
        app = graph.bulid_graph()
        return app
    
    def get_answer(self, question:str=None):
        app = self.get_graph()
        result = app.invoke({"question": question,"chat_history": self.chat_history}) 
        answer =  result['generation']
        answer = re.sub(r"\\n", '<br/>', answer)
        self.chat_history.append((question,answer)) #更新历史记录
        return "", self.chat_history  #返回本次回答和更新后的历史记录
    def clear_history(self):
        self.chat_history.clear()
        
       



def setup(model_center):
    block = gr.Blocks()

    with block as demo:
        with gr.Row(equal_height=True):           
            # gr.Image(value='../figures/1.png', scale=0.1, min_width=10, show_label=False, show_download_button=False, container=False)
    
            with gr.Column(scale=2):
                gr.Markdown("""<h1><center>智能客服</center></h1>
                    """)
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True)
                # chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True)
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="问题")

                with gr.Row():
                    # 创建提交按钮。
                    llm_btn = gr.Button("Chat with llm")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
            llm_btn.click(model_center.get_answer, inputs=[msg], outputs=[msg, chatbot])
            clear.click(model_center.clear_history)

    demo.launch()

if __name__ == "__main__":
    model_center = model_center()
    setup(model_center)
