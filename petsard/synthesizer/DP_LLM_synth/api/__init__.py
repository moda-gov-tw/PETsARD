"""
Unified Multi-Provider LLM API Library

This package provides a consistent, unified interface for interacting with
multiple Large Language Model (LLM) providers including OpenAI, Anthropic,
and Google Gemini. The design abstracts away provider-specific differences
while exposing advanced capabilities when available.

Core Components:
- APITemplate: Abstract base for all provider implementations
- ChatSession: Thread-safe session management for conversations
- ChatResult: Unified response format across all providers
- Provider implementations: OpenAIAPI, AnthropicAPI, GeminiAPI, LMStudioAPI

Key Features:
- Consistent interface across all providers
- Advanced tool/function calling with automatic execution
- Structured output support (JSON/YAML) with schema validation
- Chain-of-thought reasoning for capable models
- Thread-safe asynchronous conversation management
- Comprehensive error handling and graceful degradation
- Token usage tracking and cost optimization

Quick Start:
    from core.api import OpenAIAPI, AnthropicAPI, GeminiAPI, LMStudioAPI

    # Initialize any provider
    api = OpenAIAPI(api_key="your-openai-api-key", model="gpt-4o")
    # or
    api = AnthropicAPI(api_key="your-anthropic-api-key", model="claude-3-5-sonnet-latest")
    # or
    api = GeminiAPI(api_key="your-gemini-api-key", model="gemini-1.5-pro")
    # or
    api = LMStudioAPI(api_key="http://localhost:1234/v1", model="your-loaded-model")

    # Create a session and chat
    session = api.create_chat_session()
    session.chat("Hello, how are you?")
    result = session.wait()
    print(result.content)

    # With tools and structured output
    tools = [{"type": "function", "function": {...}}]
    session = api.create_chat_session(
        tools=tools,
        output_format="json"
    )

Provider-Specific Features:
- OpenAI: Responses API, native JSON schema, O3/O4 reasoning
- Anthropic: Thinking tokens, advanced system prompts, tool use
- Gemini: Native MIME types, multi-modal foundation, thinking config
- LMStudio: Local execution, GGUF/MLX support, grammar-based sampling, no API key

The library is designed to be easily extensible to additional providers
while maintaining backward compatibility and a consistent user experience.
"""

from .anthropic_api import AnthropicAPI
from .base import APIResponseError, APITemplate, ChatResult, ChatSession
from .gemini_api import GeminiAPI
from .lmstudio_api import LMStudioAPI
from .openai_api import OpenAIAPI

__all__ = [
    "APITemplate",
    "ChatSession",
    "ChatResult",
    "APIResponseError",
    "OpenAIAPI",
    "AnthropicAPI",
    "GeminiAPI",
    "LMStudioAPI",
]
