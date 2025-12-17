# llm/zhipu_client.py
# Factory function to get a configured Zhipu AI chat model using the official SDK.

from zai import ZhipuAiClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import Any # Import Any for arbitrary types

class CustomChatZhipuAI(BaseChatModel):
    """
    A custom wrapper for Zhipu AI's official SDK, mimicking LangChain's BaseChatModel interface.
    """
    client: Any
    model: str
    temperature: float

    def __init__(self, api_key: str, model_name: str = "glm-4.5-flash", temperature: float = 0.7, **kwargs: Any):
        # Correctly initialize Pydantic fields via super().__init__
        client = ZhipuAiClient(api_key=api_key)
        super().__init__(client=client, model=model_name, temperature=temperature, **kwargs)
        print(f"CustomChatZhipuAI: Initialized with model '{self.model}' and temperature {self.temperature}.")

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs):
        """
        Internal method to generate a chat completion.
        Converts LangChain messages to ZhipuAI format and returns a ChatResult.
        """
        zhipu_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                zhipu_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                zhipu_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                # The official SDK recommends putting system prompts in the 'messages' list with a 'system' role
                zhipu_messages.insert(0, {"role": "system", "content": message.content})
            else:
                pass # Ignore other message types for now

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=zhipu_messages,
                temperature=self.temperature,
                thinking={"type": "disabled"},
            )
            
            content = response.choices[0].message.content
            generation = ChatGeneration(message=AIMessage(content=content))
            return ChatResult(generations=[generation])

        except Exception as e:
            print(f"Error calling Zhipu AI API: {e}")
            raise e
    
    @property
    def _llm_type(self) -> str:
        return "zhipu-custom"

def get_zhipu_client(api_key: str, model_name: str = "glm-4.5-flash", temperature: float = 0.7) -> BaseChatModel:
    """
    Factory function that returns a configured Zhipu AI chat model instance.
    """
    return CustomChatZhipuAI(api_key=api_key, model_name=model_name, temperature=temperature)
