from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from dotenv import load_dotenv

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

chat_history=[
    SystemMessage(content="you are a helpful ai assistant"),
    
]

while True:
    user_input = input("you: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI:',result.content)
    
print(chat_history)