from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOllama(
    model="tinyllama",
    temperature=0,
    base_url="http://192.168.1.104:11434"
)

messages = [
    ("system", "You are an expert in {domain}. Your task is answer the question as short as possible"),
    MessagesPlaceholder("history"),
]

prompt_template = ChatPromptTemplate(messages)
domain = input('Choice domain area: ')
history = []

while True:
    print("\n"*2)
    user_content = input('User: ')
    history.append(HumanMessage(content=user_content))
    prompt_value = prompt_template.invoke({"domain": domain, "history": history})
    fill_ai_content = ""
    for ai_message_chunk in llm.stream(prompt_value):
        print(ai_message_chunk.content, end="")
        fill_ai_content += ai_message_chunk.content
    history.append(AIMessage(content=fill_ai_content))
    print()
        