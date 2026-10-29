from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser 
from langchain_classic.output_parsers import StructuredOutputParser , ResponseSchema



load_dotenv()


model = ChatOllama(model="llama3.1:latest", 
                temperature=0.1 ,
                num_gpu=24)

schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser 

# prompt = template.invoke({'topic':'black hole'})

result = chain.invoke({'topic':'black hole'})

print(result)
