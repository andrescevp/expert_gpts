# This is a sample Python script.
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessage

from mygpt.llms.openai import OpenAIApiManager

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
load_dotenv()

main_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="I am a generalist assistant. I can help you with anything.",
            name="LangAssistant",
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press Ctrl+F8 to toggle the breakpoint.

    api = OpenAIApiManager()
    answer = api.create_chat_completion(
        main_template.format_messages(text="What is the capital of spain?")
    )
    print(answer)
    tools = [
        Tool(
            name="GeneralAssistant",
            func=lambda q: api.create_chat_completion(
                main_template.format_messages(text=q)
            ),
            description="useful for when you want to answer questions that other tools can not help. The "
            "input of this tool is the question you want to answer.",
            return_direct=True,
        ),
    ]
    mem = ConversationBufferMemory(memory_key="chat_history")
    answer = api.create_chat_completion_with_agent(
        "What is the capital of spain?",
        "conversational-react-description",
        tools=tools,
        memory=mem,
    )
    emb = api.create_embedding("What is the capital of spain?")
    print(answer)
    print(emb)
    print(api.total_cost)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print_hi("PyCharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
