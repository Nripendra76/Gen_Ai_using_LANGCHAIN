from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser 
from langchain_classic.schema.runnable import RunnableParallel , RunnableBranch , RunnableLambda
from pydantic import BaseModel , Field
from typing import Literal

model = ChatOllama(model="llama3.1:latest", 
                temperature=0.1 ,
                num_gpu=22
                # num_ctx=2048
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive' , 'negative'] = Field(description='Give the sentiment of the feedback')
parser_2 = PydanticOutputParser(pydantic_object=Feedback)

prompt_1 = PromptTemplate(
    template='classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser_2.get_format_instructions()}
    
)

classifier_chain = prompt_1 | model | parser_2

prompt_2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt_3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt_2 | model | parser),
    (lambda x:x.sentiment=='negative', prompt_3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)


chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':"this is good phone"}))