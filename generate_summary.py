from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from model import llm
from prompts.summary_prompt import summary_prompt

def generate_summary(context):
    prompt_summary = summary_prompt
    prompt_qd = PromptTemplate(template=prompt_summary, input_variables=["context"])
    chain_qd = LLMChain(llm=llm, prompt=prompt_qd)
    response = chain_qd.invoke(input={"context": context})
    response = response["text"]
    return response
