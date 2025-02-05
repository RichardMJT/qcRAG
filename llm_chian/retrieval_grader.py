from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from model_to_llm import get_llm


# Data model
class GradeDocuments(BaseModel):
    """一个评分二元评分结果，用于判断问题与文档是否相关"""

    binary_score: str = Field(
        description="文档是否问问题相关的评价，值为'yes' or 'no'，相关为'yes',不相关为'no'"
    )


## 返回retrieval_grader
def get_retrieval_grader(model, temperature, api_key):
    # LLM with function call
    llm = get_llm(model=model, temperature=temperature, api_key=api_key)
    structured_llm_grader = llm.bind_tools([GradeDocuments])

    # Prompt
    # system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    #     If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    #     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    system = """你是评估检索到的文档与用户问题相关性的评分员。如果文档包含与问题相关的关键词或语义内容，则将其评为相关。\n给出一个二元评分“yes”或“no”，以表明文档是否与问题相关。"""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "文档: \n\n {document} \n\n 问题: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader