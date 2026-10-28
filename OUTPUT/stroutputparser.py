from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt1= template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result1 = model.invoke(prompt2)

print(result.content)