import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config.settings import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, get_api_key

def get_llm(
    provider: str = None, 
    model_name: str = None, 
    api_key: str = None, 
    temperature: float = 0.2
):
    """
    Factory function to retrieve a LangChain-compatible Chat LLM.
    Supports Gemini and OpenAI.
    """
    if not provider:
        provider = DEFAULT_LLM_PROVIDER
    provider = provider.lower()
    
    if not api_key:
        api_key = get_api_key(provider)
        
    if not api_key:
        # Check standard environment variables directly if configuration didn't find them
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError(
            f"API key for {provider.upper()} is not set. "
            "Please configure it in the environment, the .env file, or the Streamlit sidebar."
        )

    if provider == "gemini":
        if not model_name:
            model_name = "gemini-3.5-flash" if "flash" in DEFAULT_LLM_MODEL else "gemini-3.5-pro"
        
        # Initialize Google GenAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        
    elif provider == "openai":
        if not model_name:
            model_name = DEFAULT_LLM_MODEL if "gpt" in DEFAULT_LLM_MODEL else "gpt-4o-mini"
            
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=temperature
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
