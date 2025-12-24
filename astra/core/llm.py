"""AstraLLM - Unified LLM interface based on OpenAI API"""

import os
from typing import Literal, Optional, Iterator
from openai import OpenAI

from .exceptions import AstraException

# Supported LLM providers
SUPPORTED_PROVIDERS = Literal[
    "openai", "deepseek", "qwen", "modelscope",
    "kimi", "zhipu", "ollama", "vllm", "local", "auto"
]


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
            provider: LLM provider, auto-detected if not provided
            temperature: Temperature parameter
            max_tokens: Max token count
            timeout: Timeout in seconds, from LLM_TIMEOUT env, default 60s
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
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
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        # Check API key format
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        if actual_api_key:
            actual_key_lower = actual_api_key.lower()
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            elif actual_key_lower == "ollama":
                return "ollama"
            elif actual_key_lower == "vllm":
                return "vllm"
            elif actual_key_lower == "local":
                return "local"
            elif actual_api_key.endswith(".") or "." in actual_api_key[-20:]:
                return "zhipu"

        # Check base_url patterns
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api.openai.com" in base_url_lower:
                return "openai"
            elif "api.deepseek.com" in base_url_lower:
                return "deepseek"
            elif "dashscope.aliyuncs.com" in base_url_lower:
                return "qwen"
            elif "api-inference.modelscope.cn" in base_url_lower:
                return "modelscope"
            elif "api.moonshot.cn" in base_url_lower:
                return "kimi"
            elif "open.bigmodel.cn" in base_url_lower:
                return "zhipu"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                if ":11434" in base_url_lower or "ollama" in base_url_lower:
                    return "ollama"
                elif ":8000" in base_url_lower and "vllm" in base_url_lower:
                    return "vllm"
                elif ":8080" in base_url_lower or ":7860" in base_url_lower:
                    return "local"
                else:
                    if actual_api_key and actual_api_key.lower() == "ollama":
                        return "ollama"
                    elif actual_api_key and actual_api_key.lower() == "vllm":
                        return "vllm"
                    else:
                        return "local"
            elif any(port in base_url_lower for port in [":8080", ":7860", ":5000"]):
                return "local"

        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """Resolve API key and base_url based on provider"""
        if self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
            return resolved_api_key, resolved_base_url

        elif self.provider == "qwen":
            resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolved_api_key, resolved_base_url

        elif self.provider == "kimi":
            resolved_api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.moonshot.cn/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "zhipu":
            resolved_api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
            return resolved_api_key, resolved_base_url

        elif self.provider == "ollama":
            resolved_api_key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
            resolved_base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "vllm":
            resolved_api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "vllm"
            resolved_base_url = base_url or os.getenv("VLLM_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "local":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or "local"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        else:
            # auto or other cases: use generic config
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
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
            "openai": "gpt-4",
            "deepseek": "deepseek-chat",
            "qwen": "qwen-plus",
            "modelscope": "Qwen/Qwen2.5-72B-Instruct",
            "kimi": "moonshot-v1-8k",
            "zhipu": "glm-4",
            "ollama": "llama3.2",
            "vllm": "meta-llama/Llama-2-7b-chat-hf",
            "local": "local-model",
        }
        
        if self.provider in defaults:
            return defaults[self.provider]
        
        # Auto-detect from base_url
        base_url = os.getenv("LLM_BASE_URL", "").lower()
        if "modelscope" in base_url:
            return "Qwen/Qwen2.5-72B-Instruct"
        elif "deepseek" in base_url:
            return "deepseek-chat"
        elif "dashscope" in base_url:
            return "qwen-plus"
        elif "moonshot" in base_url:
            return "moonshot-v1-8k"
        elif "bigmodel" in base_url:
            return "glm-4"
        elif "ollama" in base_url or ":11434" in base_url:
            return "llama3.2"
        elif ":8000" in base_url or "vllm" in base_url:
            return "meta-llama/Llama-2-7b-chat-hf"
        elif "localhost" in base_url or "127.0.0.1" in base_url:
            return "local-model"
        
        return "gpt-4"

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
