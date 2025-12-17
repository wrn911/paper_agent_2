# agent/wrapper.py
# This file wraps the LangChain agent, abstracting away the details of its setup and execution.

import tiktoken
import logging
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

# Configuration and clients
from config.settings import Settings
from llm.client import get_llm_client
from utils.logger import log_token_usage # Import token logger
from config import prompts

# Tools
from tools.file_tools import write_file

@tool
def file_writer_tool(file_path: str, content: str) -> str:
    """A tool to write files."""
    return write_file(file_path, content)


class LangChainWrapper:
    """
    A wrapper for a LangChain agent that simplifies its initialization and execution.
    This wrapper now manages the agent's memory and manually logs token usage.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initializes the LLM and tools.
        """
        self.logger = logger
        settings = Settings()
        provider = settings.get('llm', {}).get('provider', 'openai')
        api_keys = settings.get('api_keys', {})

        if provider == 'openai':
            api_key = api_keys.get('openai')
            if not api_key or api_key == "YOUR_CHATGPT_API_KEY":
                raise ValueError("OpenAI API key not configured in config/config.yaml")
            
            base_url = settings.get('llm', {}).get('openai_base_url')
            if not base_url:
                raise ValueError("OpenAI base_url not configured under 'llm' in config/config.yaml")
            
            self.model_name = "gpt-5.1" # TODO: Move to config
            self.llm = get_llm_client(api_key=api_key, base_url=base_url, model_name=self.model_name)
            self.logger.info("Initialized with OpenAI LLM (custom endpoint).")
            # Initialize tiktoken encoding for the specified model
            try:
                self.encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                self.logger.warning(f"Model '{self.model_name}' not found for tiktoken. Using 'cl100k_base' encoding.")
                self.encoding = tiktoken.get_encoding("cl100k_base")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        self.tools = [file_writer_tool]
        self.logger.info("Tools loaded (File Write only).")

        self.chat_history = []
        self._initialize_agent_executor()

    def _initialize_agent_executor(self, system_prompt: str = ""):
        if not system_prompt:
            system_prompt = prompts.SYSTEM_PROMPT
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.llm, self.tools, prompt_template)
        # verbose=True will now print to the configured root logger, including thread name
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def _count_tokens(self, text: str) -> int:
        """Counts tokens in a string using the initialized tiktoken encoding."""
        return len(self.encoding.encode(text))

    def run(self, user_prompt: str, system_prompt: str = "") -> str:
        """
        Executes the LangChain Agent, manually calculates token usage, and logs everything.
        """
        self.logger.info(f"Invoking LLM with User Prompt:\n{user_prompt}")
        
        if system_prompt:
            self._initialize_agent_executor(system_prompt)
        
        # Manually calculate prompt tokens
        prompt_tokens = self._count_tokens(system_prompt) + self._count_tokens(user_prompt)
        for msg in self.chat_history:
            prompt_tokens += self._count_tokens(msg.content)

        # The AgentExecutor's verbose output will be handled by the root logger setup in main.
        result = self.agent_executor.invoke(
            {"input": user_prompt, "chat_history": self.chat_history}
        )
        
        output_content = result.get("output", "")
        
        # Manually calculate completion tokens
        completion_tokens = self._count_tokens(output_content)
        
        # Log locally calculated token usage
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
        self.logger.info(f"[TOKEN USAGE] Model: {self.model_name}, "
                         f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")
        log_token_usage(self.model_name, token_usage)
        
        self.chat_history.append(HumanMessage(content=user_prompt))
        self.chat_history.append(AIMessage(content=output_content))
        
        return output_content

    def reset_memory(self):
        """
        Resets the agent's chat history.
        """
        self.logger.info("Resetting agent memory.")
        self.chat_history = []
        self._initialize_agent_executor()
