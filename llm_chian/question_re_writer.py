from model_to_llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Data model
# class QuestionDocument(BaseModel):
#     """中文字符串格式的重写后的问题"""

#     reQuestion: str = Field(
#         description="重写后的问题，是一个中文字符串"
#     )


### Question Re-writer
def get_question_rewriter(model, temperature, api_key):

    # LLM
    llm = get_llm(model, temperature, api_key)

    # llm.bind_tools([QuestionDocument])

    # Prompt
    system = """你首先要判断输入是否为问题，如果不是问题，不修改原始问题，直接输出原始输入，不输出其他内容。\n
    如果是问题，请分析其潜在含义并根据含义重写问题，重写后的问题将被用于在搜索引擎搜索答案，只输出重写后的问题，不输出其他内容。\n """
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "这是原始输入 \n\n {question} \n ",
            ),
        ]
    )

    # question_rewriter = re_write_prompt | llm | StrOutputParser()
    question_rewriter = re_write_prompt | llm
    return question_rewriter
