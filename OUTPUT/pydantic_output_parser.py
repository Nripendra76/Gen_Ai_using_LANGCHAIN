from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser , PydanticOutputParser
from langchain_classic.output_parsers import StructuredOutputParser , ResponseSchema
from pydantic import BaseModel , Field


load_dotenv()


model = ChatOllama(model="llama3.1:latest", 
                temperature=0.1 ,
                num_gpu=24
                # num_ctx=2048
)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18 , description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")
    
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name ,age and the city of  a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':"indian"})

print(result)
print(template)