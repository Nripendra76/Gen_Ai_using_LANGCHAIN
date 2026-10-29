from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3.1:latest", 
                temperature=0.1 ,
                num_gpu=24
                # num_ctx=2048
)

prompt = PromptTemplate(
    template="how many types of chain are there in langchain",
    input_variables=[]
)
# Generate 5 interesting facts about {topic}   'topic': "basketball"
parser = StrOutputParser()

chain = prompt | model | parser

result=chain.invoke({})

print(result)

chain.get_graph().print_ascii()