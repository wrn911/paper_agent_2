# llm/client.py
# Factory function to get a configured LangChain chat model (OpenAI-style or Gemini).

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def get_llm_client(
    api_key: str,
    model_name: str = "gpt-5.1",
    temperature: float = 0.7,
    base_url: str = None,
    provider: str = "openai",
) -> BaseChatModel:
    """
    Factory function that returns a configured LangChain Chat object.

    Args:
        api_key (str): API key for the LLM service.
        model_name (str): Model name (e.g., "gpt-5.1" or "gemini-1.5-pro").
        temperature (float): Sampling temperature.
        base_url (str): Optional custom base URL for OpenAI-compatible APIs.
        provider (str): "openai" or "gemini".

    Returns:
        BaseChatModel: A configured LangChain chat model instance.
    """
    provider = (provider or "openai").lower()

    if provider == "openai":
        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            streaming=False,  # Explicitly disable streaming
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:  # pragma: no cover - runtime guard
            raise ImportError("Please install langchain-google-genai to use Gemini models.") from e

        return ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {provider}")
