import os
import sys
absPath = os.path.abspath(__file__)   #返回代码段所在的位置，肯定是在某个.py文件中
temPath01 = os.path.dirname(absPath)    #往上返回一级目录，得到文件所在的路径
temPath = os.path.dirname(temPath01)    #在往上返回一级，得到文件夹所在的路径
sys.path.append(temPath01)   
sys.path.append(temPath)   

from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from database.get_vectordb import get_vectordb
from langgraph.graph import END, StateGraph, START
from llm_chian.rag_chain import get_rag_chain
from llm_chian.retrieval_grader import get_retrieval_grader
from llm_chian.question_re_writer import get_question_rewriter
from tool.search import get_web_search_tool




class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

class GraphPoint():
    '''
        创建图中的结点
    '''
    # def __init__(self, api_key, search_type="similarity", search_kwargs={'k': 2},model = "qwen-max", temperature = 0) -> None:
    def __init__(self,retriever, rag_chain, retrieval_grader, question_rewriter, web_search_tool) -> None:
        """
            Args:
               
        """
        file_path = "../knowledge_db"
        persist_path = "../vector_db/chroma"
        ## 创建向量数据库
        # vectordb = get_vectordb(file_path, persist_path)
        ## 创建检索器
        # self.retriever = vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        # self.rag_chain = get_rag_chain(model = model, temperature = temperature, api_key = api_key)
        # self.retrieval_grader = get_retrieval_grader(model = model, temperature = temperature, api_key = api_key)
        # self.question_rewriter = get_question_rewriter(model = model, temperature = temperature, api_key = api_key)
        # self.web_search_tool = get_web_search_tool()

        self.retriever = retriever
        self.rag_chain = rag_chain
        self.retrieval_grader = retrieval_grader
        self.question_rewriter = question_rewriter
        self.web_search_tool = web_search_tool
    
    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            # print("score")
            # print(score.tool_calls[0]['args']['binary_score'])
            if score :
                # grade = score.binary_score 使用在线大模型时的调用方法
                grade = score.tool_calls[0]['args']['binary_score'] #使用chatollama时的获取方法
            else:
                grade = None
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}


    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        # better_question = better_question.content #这是使用chatollama绑定工具后输出的格式， 如果更换别的大模型，可能需要更换代码编写方法
        print('---------------------------------')
        print(better_question)
        print('---------------------------------')
        better_question = better_question.content
        return {"documents": documents, "question": better_question}


    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """
        
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        print("-----------question------------")
        print( question)
        print("-----------question------------")
        # Web search
        
        docs = self.web_search_tool.run(question)
        web_results = "\n".join([d for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "question": question}


    ### Edges


    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
        
    def bulid_graph(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("web_search_node", self.web_search)  # web search

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)

        # Compile
        app = workflow.compile()

        return app
    
    ## 返回向量数据库中的检索器
    # def get_retriever(self):
    #     return self.retriever