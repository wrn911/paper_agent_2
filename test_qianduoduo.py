from langchain_openai import ChatOpenAI
import os


llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
    base_url="https://api2.aigcbest.top/v1",
    api_key="sk-1OUmm4rEXt4Hk3eB4HxWrgBD9ImjOINn9pxXIEx6rwxm68QR"
)

print(llm.invoke("Say 'test from qianduoduo'").content)
