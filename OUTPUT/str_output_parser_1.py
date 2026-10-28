from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm= HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

template1= PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)


template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'black hole'})

print(result)