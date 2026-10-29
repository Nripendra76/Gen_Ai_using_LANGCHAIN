from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3.1:latest", 
                temperature=0.1 ,
                num_gpu=24
                # num_ctx=2048
)

prompt1= PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
    
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser= StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()