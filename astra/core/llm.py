"""AstraLLM - Unified LLM interface based on OpenAI API"""

import os
from typing import Literal, Optional, Iterator
from openai import OpenAI

from .exceptions import AstraException

# Supported LLM providers
SUPPORTED_PROVIDERS = Literal["bedrock", "openai", "ollama", "vllm", "auto"]


class AstraLLM:
    """
    Astra LLM client for calling any OpenAI-compatible service.
    
    Design philosophy:
    - Parameters first, environment variables as fallback
    - Streaming response as default for better UX
    - Multi-provider support
    - Unified calling interface
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the client. Parameters take priority, environment variables as fallback.

        Args:
            model: Model name, reads from LLM_MODEL_ID env if not provided
            api_key: API key, reads from environment if not provided
            base_url: Service URL, reads from LLM_BASE_URL env if not provided
            provider: LLM provider (bedrock, openai, ollama, vllm), auto-detected if not provided
            temperature: Temperature parameter
            max_tokens: Max token count
            timeout: Timeout in seconds, from LLM_TIMEOUT env, default 60s
        """
        self.model = model or os.getenv("LLM_MODEL_ID") or os.getenv("BEDROCK_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        # Auto-detect or use specified provider
        self.provider = provider or self._auto_detect_provider(api_key, base_url)

        # Resolve API key and base_url based on provider
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        # Validate required parameters
        if not self.model:
            self.model = self._get_default_model()
        if not all([self.api_key, self.base_url]):
            raise AstraException("API key and base URL must be provided or defined in .env file.")

        # Create OpenAI client
        self._client = self._create_client()

    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """Auto-detect LLM provider based on environment and parameters"""
        # Check provider-specific env variables
        if os.getenv("AWS_BEDROCK_API_KEY") or os.getenv("BEDROCK_MODEL_ID"):
            return "bedrock"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        # Check base_url patterns
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "bedrock" in base_url_lower or "amazonaws.com" in base_url_lower:
                return "bedrock"
            elif "api.openai.com" in base_url_lower:
                return "openai"
            elif "localhost:11434" in base_url_lower or "ollama" in base_url_lower:
                return "ollama"
            elif "localhost:8000" in base_url_lower or "vllm" in base_url_lower:
                return "vllm"

        # Default to bedrock
        return "bedrock"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """Resolve API key and base_url based on provider"""
        if self.provider == "bedrock":
            resolved_api_key = api_key or os.getenv("AWS_BEDROCK_API_KEY") or os.getenv("LLM_API_KEY") or "bedrock"
            aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
            resolved_base_url = base_url or os.getenv("BEDROCK_BASE_URL") or os.getenv("LLM_BASE_URL") or f"https://bedrock-runtime.{aws_region}.amazonaws.com"
            return resolved_api_key, resolved_base_url

        elif self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "ollama":
            resolved_api_key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
            resolved_base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "vllm":
            resolved_api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "vllm"
            resolved_base_url = base_url or os.getenv("VLLM_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        else:
            # auto: use generic config, default to bedrock
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or "bedrock"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            if not resolved_base_url:
                aws_region = os.getenv("AWS_REGION") or "us-east-1"
                resolved_base_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com"
            return resolved_api_key, resolved_base_url

    def _create_client(self) -> OpenAI:
        """Create OpenAI client"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def _get_default_model(self) -> str:
        """Get default model for provider"""
        defaults = {
            "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0",
            "openai": "gpt-4",
            "ollama": "llama3.2",
            "vllm": "meta-llama/Llama-2-7b-chat-hf",
        }
        return defaults.get(self.provider, "anthropic.claude-3-sonnet-20240229-v1:0")

    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        Call LLM with streaming response.
        
        Args:
            messages: Message list
            temperature: Temperature parameter

        Yields:
            str: Streaming response text chunks
        """
        print(f"🧠 Calling {self.model} model...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            print("✅ LLM response:")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()

        except Exception as e:
            print(f"❌ Error calling LLM API: {e}")
            raise AstraException(f"LLM call failed: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        Non-streaming LLM call, returns complete response.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            return response.choices[0].message.content
        except Exception as e:
            raise AstraException(f"LLM call failed: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """Streaming LLM call alias method."""
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)
