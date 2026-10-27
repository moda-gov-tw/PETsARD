"""
Base API template and chat session infrastructure with unified I/O.

This module provides the foundational architecture for a unified multi-provider
LLM API system. It abstracts away provider-specific differences while exposing
common capabilities across different LLM providers (OpenAI, Anthropic, Google).

Key Components:
- APITemplate: Abstract base class that all provider APIs inherit from
- ChatSession: Thread-safe chat session management with async capabilities
- ChatResult: Unified response format across all providers
- Utility functions: Tool normalization, output parsing, function execution

Supported Features:
- Tool/Function calling with automatic execution
- Structured outputs (JSON/YAML) with schema validation
- Chain-of-thought (COT) reasoning when supported by the model
- Thread-safe session management
- Unified error handling and response formats

Design Philosophy:
The design prioritizes consistency and extensibility. Each provider implements
the same interface but can expose provider-specific features through parameters.
This allows users to switch between providers with minimal code changes while
still accessing advanced features when needed.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

JSONDict = dict[str, Any]


def _try_parse_json(content: str) -> tuple[Any | None, Exception | None]:
    """Attempt to parse a string as JSON.

    Args:
        content: The string content to parse as JSON

    Returns:
        A tuple of (parsed_object, error). If parsing succeeds, returns
        (parsed_data, None). If parsing fails, returns (None, exception).

    Note:
        This function is designed to be safe and never raises exceptions.
        Instead, it returns the exception for the caller to handle or ignore.
    """
    try:
        return json.loads(content), None
    except Exception as exc:  # noqa: BLE001 - return parse error for introspection
        return None, exc


def _try_parse_yaml(content: str) -> tuple[Any | None, Exception | None]:
    """Attempt to parse a string as YAML.

    Args:
        content: The string content to parse as YAML

    Returns:
        A tuple of (parsed_object, error). If parsing succeeds, returns
        (parsed_data, None). If parsing fails, returns (None, exception).

    Note:
        This function requires the PyYAML library. If the library is not
        available, it will return an ImportError as the exception.
        Like _try_parse_json, this function never raises exceptions.
    """
    try:
        import yaml  # type: ignore

        return yaml.safe_load(content), None
    except Exception as exc:  # noqa: BLE001 - return parse error for introspection
        return None, exc


class APIResponseError(RuntimeError):
    """Raised when a provider reports a failure inside an otherwise successful response.

    Some providers (OpenAI-compatible gateways in particular) answer with HTTP 200
    and put the failure in the response body instead of raising, e.g.
    ``Response(status="failed", error=ResponseError(code=..., message=...), output=[])``.
    Such "soft" errors would otherwise surface as an empty ChatResult.

    Attributes:
        provider: Name of the provider that returned the response.
        message: Error message reported by the provider.
        code: Provider error code/type when available (e.g. "server_error").
        status: Response status when available (e.g. "failed").
        model: Model the request was sent to.
        response: The raw response object, for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        code: str | None = None,
        status: str | None = None,
        model: str | None = None,
        response: Any = None,
    ) -> None:
        self.provider = provider
        self.message = message
        self.code = code
        self.status = status
        self.model = model
        self.response = response

        response_id = _get_field(response, "id") or _get_field(response, "response_id")
        details = ", ".join(
            f"{key}={value}"
            for key, value in (
                ("code", code),
                ("status", status),
                ("model", model),
                ("response_id", response_id),
            )
            if value
        )
        super().__init__(
            f"[{provider}] API response error"
            + (f" ({details})" if details else "")
            + f": {message}"
        )


def _get_field(obj: Any, name: str) -> Any:
    """Read a field from an SDK object or a plain dict; None if missing."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def raise_for_response_error(
    response: Any,
    *,
    provider: str,
    model: str | None = None,
) -> None:
    """Raise APIResponseError if the response carries an error payload.

    Detects "soft" errors where the SDK call returned normally but the response
    body reports a failure: a non-empty ``error`` field, ``type == "error"``,
    or a ``status`` of "failed"/"cancelled".

    Args:
        response: The raw provider response (SDK object or dict)
        provider: Provider name used in the error message
        model: Model name used in the error message

    Raises:
        APIResponseError: If the response reports a failure.

    Note:
        ``status == "incomplete"`` (e.g. max_output_tokens reached) is not
        treated as an error because the response still carries usable output.
    """
    error = _get_field(response, "error")
    status = _get_field(response, "status")
    status = status if isinstance(status, str) else None
    is_error_type = _get_field(response, "type") == "error"

    if not error and not is_error_type and status not in {"failed", "cancelled"}:
        return

    if error and not isinstance(error, str):
        code = _get_field(error, "code") or _get_field(error, "type")
        message = _get_field(error, "message") or str(error)
    else:
        code = None
        message = error or f"response status is '{status or 'error'}' without error details"

    raise APIResponseError(
        str(message),
        provider=provider,
        code=str(code) if code else None,
        status=status,
        model=model,
        response=response,
    )


@dataclass
class ChatResult:
    """Unified chat result across providers.

    This class provides a consistent response format regardless of which
    LLM provider is used. It normalizes the different response formats
    from OpenAI, Anthropic, Google, etc. into a single structure.

    Attributes:
        content: The main text content from the assistant's response.
            Even when structured output is requested and parsing succeeds,
            this contains the raw string form for debugging purposes.

        parsed: The parsed structured output when output_format is "json" or "yaml"
            and parsing succeeded. None if parsing failed or format was "text".

        format: The requested output format ("text" | "json" | "yaml").
            This reflects what was requested, not necessarily what was delivered.

        tool_calls: List of tool/function calls made by the assistant.
            Each tool call is normalized to a provider-agnostic format.
            None if no tools were called.

        usage: Token usage and cost metadata when available from the provider.
            Format varies by provider but typically includes prompt_tokens,
            completion_tokens, and total_tokens.

        usage_input_token: The number of input tokens used by the model.

        usage_cache_input_token: The number of cached input tokens used by the model.
            This represents tokens that were retrieved from cache rather than processed.
            For providers without caching (e.g., LM Studio), this will be 0.

        usage_output_token: The number of output tokens used by the model.

        raw: The original, unprocessed response object from the provider.
            Useful for debugging or accessing provider-specific features
            not exposed in the normalized format.

        reasoning: Optional metadata about the model's reasoning process.
            When models support chain-of-thought or thinking modes,
            their reasoning is exposed under reasoning["thinking_blocks"]
            as a list of strings.

        output: Raw list of all output elements from the provider.
            This includes the original response plus any tool interaction
            messages that were part of the conversation flow.

        id: The id of the chat session.
    """

    content: str
    parsed: Any | None
    format: str
    tool_calls: list[JSONDict] | None
    usage: JSONDict | None
    usage_input_token: int
    usage_cache_input_token: int
    usage_output_token: int
    raw: Any
    reasoning: JSONDict | None
    output: list[Any] | None
    id: str


class ChatSession:
    """A threaded chat session that calls into a provider via APITemplate.

    This class manages a conversational session with an LLM provider,
    handling message history, threading, and configuration. It provides
    an asynchronous interface where you can start a request and wait
    for completion, allowing for non-blocking operations.

    Key Features:
    - Thread-safe message history management
    - Asynchronous request handling with .chat() and .wait()
    - Automatic conversation continuity
    - Support for tools, structured output, and chain-of-thought
    - Provider-agnostic interface

    Usage:
        # Basic usage
        session = api.create_chat_session()
        session.chat("Hello, how are you?")
        result = session.wait()
        print(result.content)

        # With structured output
        session = api.create_chat_session(output_format="json")
        session.chat("Give me a JSON list of colors")
        result = session.wait()
        colors = result.parsed  # Parsed JSON object

        # With tools
        tools = [{"type": "function", "function": {...}}]
        session = api.create_chat_session(tools=tools)
        session.chat("What's the weather like?")
        result = session.wait()  # Tool will be called automatically

    Thread Safety:
        This class is thread-safe. Multiple threads can call methods
        safely, but only one request can be in progress at a time.
        Calling chat() while another request is active will raise RuntimeError.
    """

    def __init__(
        self,
        api: APITemplate,
        system_prompt: str | None,
        tools: list[JSONDict] | None = None,
        output_format: str = "text",
        json_schema: JSONDict | None = None,
        enable_cot: bool = False,
        session_params: JSONDict | None = None,
        custom_messages: list[JSONDict] | None = None,
    ) -> None:
        """Initialize a new chat session.

        Args:
            api: The APITemplate instance to use for requests
            system_prompt: Optional system prompt to set the assistant's behavior
            tools: List of tool/function definitions for the assistant to use
            output_format: Desired output format ("text", "json", or "yaml")
            json_schema: Optional JSON schema for structured output validation
            enable_cot: Whether to enable chain-of-thought reasoning if supported
            session_params: Provider-specific session parameters
            custom_messages: Pre-populate the session with custom message history
        """
        self._api = api
        self._system_prompt = system_prompt
        self._tools = tools or []
        self._output_format = output_format
        self._json_schema = json_schema
        self._enable_cot = enable_cot
        self._session_params = api._translate_request_params(dict(session_params or {}))

        # Initialize message history
        self._messages: list[JSONDict] = custom_messages if custom_messages else [{"role": "system", "content": system_prompt}] if system_prompt else []

        # Threading infrastructure for async operations
        self._lock = threading.Lock()  # Protects shared state
        self._thread: threading.Thread | None = None  # Current request thread
        self._last_result: ChatResult | None = None  # Last successful result
        self._last_error: BaseException | None = None  # Last error, if any

    # ------------------------------ public API ------------------------------ #
    def chat(
        self,
        user_message: str,
        tool_choice: str | JSONDict | None = None,
        request_params: JSONDict | None = None,
    ) -> None:
        """Send a message and start a background request thread.

        This method starts an asynchronous request to the LLM provider.
        The actual API call happens in a background thread, allowing the
        calling code to continue executing. Use wait() to block until
        the response is ready.

        Args:
            user_message: The message to send to the assistant
            tool_choice: Optional tool selection hint ("auto", "none", or specific tool)
            request_params: Per-request parameters in unified names (see
                APITemplate._translate_request_params). Common keys:
                max_tokens, temperature, top_p, top_k.

        Raises:
            RuntimeError: If a request is already in progress. Call wait() first.

        Note:
            This method is non-blocking. The actual API call happens in a
            background thread. Any exceptions during the API call will be
            captured and re-raised when wait() is called.
        """
        with self._lock:
            # Add user message to conversation history
            _messages = self.api._prepare_user_message(user_message)

            # Ensure no other request is in progress
            if self._thread and self._thread.is_alive():
                raise RuntimeError("A request is already in progress; wait() first")

            # Reset previous results
            self._last_result = None
            self._last_error = None

            def _target() -> None:
                """Background thread target that performs the actual API call."""
                try:
                    # Make the API request through the provider
                    result = self._api._send_messages(  # noqa: SLF001 - intentional
                        messages=self._messages+_messages,
                        tools=self._tools,
                        tool_choice=tool_choice,
                        output_format=self._output_format,
                        json_schema=self._json_schema,
                        enable_cot=self._enable_cot,
                        request_params=self._api._translate_request_params(
                            dict(request_params or {})
                        ),
                        session_params=self._session_params,
                    )
                    # Append assistant's response to conversation for continuity
                    # Only if the assistant successfully response, add the message to history
                    self._messages += _messages
                    self._messages.extend(result.output)
                    self._last_result = result
                except BaseException as exc:  # noqa: BLE001 - retain raw exception
                    self._last_error = exc

            # Start the background request
            self._thread = threading.Thread(target=_target, daemon=True)
            self._thread.start()

    def wait(self, timeout: float | None = None) -> ChatResult:
        """Wait for the background request to complete and return the result.

        This method blocks until the background request started by chat()
        completes successfully or fails. If the request failed, this method
        will re-raise the original exception.

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            ChatResult: The response from the LLM provider

        Raises:
            RuntimeError: If no request is in progress (call chat() first)
            TimeoutError: If the request doesn't complete within the timeout
            Any exception raised during the API request

        Note:
            This method is safe to call multiple times. If the request has
            already completed, it will return the cached result immediately.
        """
        # Get the current thread reference safely
        thread = None
        with self._lock:
            thread = self._thread

        if thread is None:
            raise RuntimeError("No request in progress. Call chat() first.")

        # Wait for the background thread to complete
        thread.join(timeout=timeout)

        # Check the result safely
        with self._lock:
            if thread.is_alive():
                raise TimeoutError("wait() timed out before the response was ready")
            if self._last_error is not None:
                raise self._last_error
            if self._last_result is None:
                raise RuntimeError("Background thread finished without a result")
            return self._last_result

    # --------------------------- accessors/helpers -------------------------- #
    @property
    def messages(self) -> list[JSONDict]:
        """Get a copy of the current conversation history.

        Returns:
            A copy of the message list. Modifying this list will not
            affect the actual conversation history.
        """
        return list(self._messages)

    @messages.setter
    def messages(self, value: list[JSONDict]) -> None:
        """Set the conversation history."""
        self._messages = list(value)

    @property
    def api(self) -> APITemplate:
        """Get the API instance."""
        return self._api

    @api.setter
    def api(self, value: APITemplate) -> None:
        """Set the API instance."""
        self._api = value

    @property
    def system_prompt(self) -> list[JSONDict]:
        """Get the current system prompt.

        Returns:
            The current system prompt.
        """
        return self._system_prompt


class APITemplate(ABC):
    """Abstract base class for provider API wrappers.

    This is the foundation class that all LLM provider implementations
    inherit from. It defines the common interface and provides shared
    functionality like function registry management and session creation.

    Subclasses must implement `_send_messages` to handle the provider-specific
    API communication while returning a unified ChatResult. This design
    allows the system to support multiple providers with a consistent interface.

    Common Features Provided:
    - Function/tool registry for automated function execution
    - System prompt management
    - Chat session creation with consistent parameters
    - Input validation and error handling

    Provider-Specific Implementation Required:
    - Authentication and client setup
    - Message format translation
    - Tool/function call handling
    - Response parsing and normalization
    - Provider-specific feature detection (e.g., COT support)
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        api_url: str | None = None,
    ) -> None:
        """Initialize the API wrapper with authentication and model selection.

        Args:
            api_key: Authentication key for the provider's API
            model: Specific model to use. If None, subclass will use a default.
            api_url: Optional custom endpoint URL for the provider's API.
                    If None, the subclass uses its default endpoint.
        """
        self._api_key = api_key
        self._model = model
        self._api_url = api_url
        self._system_prompt: str | None = None
        # Mapping from function name to callable for custom tool/function calls
        self._function_registry: dict[str, Callable[..., Any]] = {}

    # ------------------------------ core surface ---------------------------- #
    def set_system_prompt(self, prompt: str | None) -> None:
        """Set or update the system prompt for this API instance.

        Args:
            prompt: The system prompt to use for new sessions, or None to clear it.
        """
        self._system_prompt = prompt

    def get_model(self) -> str:
        """Get the currently configured model name.

        Returns:
            The model identifier string.
        """
        return self._model

    def get_system_prompt(self) -> str | None:
        """Get the current system prompt.

        Returns:
            The current system prompt string, or None if not set.
        """
        return self._system_prompt

    # --------------------------- function registry --------------------------- #
    def set_function_registry(self, functions: dict[str, Callable[..., Any]]) -> None:
        """Replace the current function registry.

        The function registry allows the LLM to call Python functions automatically
        when tool calls are made. This enables powerful integrations where the AI
        can interact with external systems, databases, APIs, etc.

        Args:
            functions: Dictionary mapping function names to callable objects.
                      The mapping must be of the form {function_name: callable}.

        Example:
            def get_weather(city: str) -> str:
                return f"Weather in {city}: Sunny, 25°C"

            api.set_function_registry({"get_weather": get_weather})
        """
        self._function_registry = dict(functions or {})

    def register_function(self, name: str, func: Callable[..., Any]) -> None:
        """Register a single function under a given name.

        Args:
            name: The function name that the LLM will use to call this function
            func: The callable to execute when the function is called

        Raises:
            ValueError: If name is empty or func is not callable

        Example:
            def calculate_sum(a: int, b: int) -> int:
                return a + b

            api.register_function("add", calculate_sum)
        """
        if not name or not callable(func):
            raise ValueError("register_function requires a non-empty name and a callable")
        self._function_registry[name] = func

    def clear_function_registry(self) -> None:
        """Clear all registered functions.

        After calling this, the LLM will not be able to execute any
        custom functions, though it may still attempt to call them.
        """
        self._function_registry.clear()

    def create_chat_session(
        self,
        *,
        tools: list[JSONDict] | None = None,
        output_format: str = "text",
        json_schema: JSONDict | None = None,
        enable_cot: bool = False,
        session_params: JSONDict | None = None,
        custom_messages: list[JSONDict] | None = None,
    ) -> ChatSession:
        """Create a chat session bound to this API instance.

        Creates a new conversation session with the specified configuration.
        The session maintains conversation history and handles threading
        for asynchronous requests.

        Args:
            tools: List of tool/function definitions for the assistant to use.
                  Each tool should follow the standard function calling format.
            output_format: Desired response format ("text" | "json" | "yaml").
                         "json" and "yaml" will attempt structured output parsing.
            json_schema: Optional JSON schema for validating structured output.
                        Only used when output_format is "json".
            enable_cot: Request chain-of-thought reasoning if the model supports it.
                       This may expose the model's thinking process.
            session_params: Parameters for every request in this session, using
                          unified names (see APITemplate._translate_request_params).
                          Examples: max_tokens, temperature, reasoning_effort.
            custom_messages: Pre-populate the session with existing conversation history.
                           Useful for continuing conversations or setting context.

        Returns:
            ChatSession: A new session instance ready for conversation.

        Raises:
            ValueError: If output_format is not one of the supported values.

        Example:
            # Basic text conversation
            session = api.create_chat_session()

            # JSON output with schema validation
            schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
            session = api.create_chat_session(
                output_format="json",
                json_schema=schema
            )

            # With tools and reasoning
            tools = [{"type": "function", "function": {...}}]
            session = api.create_chat_session(
                tools=tools,
                enable_cot=True
            )
        """
        if output_format not in {"text", "json", "yaml"}:
            raise ValueError("output_format must be one of: text, json, yaml")

        return ChatSession(
            api=self,
            system_prompt=self._system_prompt,
            tools=tools,
            output_format=output_format,
            json_schema=json_schema,
            enable_cot=enable_cot,
            session_params=session_params,
            custom_messages=custom_messages,
        )

    # ------------------------ provider-specific wiring ---------------------- #
    @abstractmethod
    def _send_messages(
        self,
        *,
        messages: list[JSONDict],
        tools: list[JSONDict] | None,
        tool_choice: str | JSONDict | None,
        output_format: str,
        json_schema: JSONDict | None,
        enable_cot: bool,
        request_params: JSONDict,
        session_params: JSONDict,
    ) -> ChatResult:
        """Provider-specific implementation of message sending and response handling.

        This is the core method that each provider subclass must implement to
        handle the actual API communication. It should:

        1. Convert the unified message format to provider-specific format
        2. Configure tools/functions according to provider requirements
        3. Set up structured output based on provider capabilities
        4. Enable reasoning/COT features when supported and requested
        5. Make the API request with proper error handling
        6. Parse the response and extract relevant information
        7. Execute any tool calls and handle follow-up requests
        8. Return a unified ChatResult with normalized data

        Args:
            messages: Conversation history in unified format
            tools: Tool/function definitions in generic format
            tool_choice: Tool selection strategy ("auto", "none", or specific tool)
            output_format: Desired response format ("text", "json", "yaml")
            json_schema: Optional JSON schema for structured output validation
            enable_cot: Whether to enable chain-of-thought reasoning if supported
            request_params: Request-specific parameters (temperature, max_tokens, etc.)
            session_params: Session-level parameters that persist across requests

        Returns:
            ChatResult: Unified response containing content, parsed data, tool calls,
                       usage information, and any reasoning metadata

        Raises:
            Should re-raise provider-specific exceptions with minimal wrapping
            to preserve error details for debugging and handling

        Implementation Notes:
            - Handle provider authentication and client initialization errors
            - Convert tool schemas using appropriate normalization functions
            - Respect provider-specific parameter names and value ranges
            - Extract and normalize usage/billing information when available
            - Handle thinking/reasoning tokens appropriately for each provider
            - Ensure thread safety if the provider client supports it
        """

    def _prepare_user_message(self, user_message: str) -> list[JSONDict]:
        """Prepare the user message for the API.

        Args:
            user_message: The user message to prepare

        Returns:
            The prepared user message.
        """
        return [{"role": "user", "content": user_message}] if user_message else [{"role": "user", "content": "continue"}]

    def _translate_request_params(
        self,
        request_params: JSONDict,
    ) -> JSONDict:
        """Map unified chat/session parameters to this provider's API names.

        Callers should use provider-agnostic names in ``request_params`` and
        ``session_params`` (notably ``max_tokens`` for output length). Each
        subclass overrides this to rename keys before ``_send_messages()``.

        Unified keys (when supported by the provider):
            max_tokens, temperature, top_p, top_k,
            reasoning_effort (OpenAI / LM Studio; read in _send_messages),
            thinking_budget_tokens (Anthropic), thinking_budget (Gemini).

        Default implementation returns a shallow copy unchanged. Subclasses
        implement provider-specific aliases (e.g. max_tokens → max_output_tokens
        for OpenAI Responses API and Gemini).
        """
        return dict(request_params)


# --------------------------- utility mixin/helpers -------------------------- #

def normalize_tool_schemas_for_openai(tools: list[JSONDict]) -> list[JSONDict]:
    """Normalize generic tool schemas to OpenAI tool format if needed.

    OpenAI expects tools in a specific format with type and function fields.
    This function ensures tools conform to that format while preserving
    compatibility with tools that are already correctly formatted.

    Expected generic format:
        {
            "type": "function",
            "function": {
                "name": str,
                "description": str,
                "parameters": JSONSchema
            }
        }

    Args:
        tools: List of tool definitions in various formats

    Returns:
        List of tools normalized to OpenAI format

    Note:
        This function is primarily for compatibility. Most tools should
        already be in the correct format when passed to the API.
    """
    normalized: list[JSONDict] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            # Chat Completions nested format → Responses API flat format
            fn = tool["function"]
            flat_tool: JSONDict = {
                "type": "function",
                "name": fn["name"],
            }
            if "description" in fn:
                flat_tool["description"] = fn["description"]
            if "parameters" in fn:
                flat_tool["parameters"] = fn["parameters"]
            normalized.append(flat_tool)
        elif tool.get("type") == "function" and "name" in tool:
            # Already in Responses API flat format
            normalized.append(tool)
        else:
            normalized.append(tool)
    return normalized


def normalize_tool_schemas_for_lmstudio(tools: list[JSONDict]) -> list[JSONDict]:
    """Normalize tool schemas to the OpenAI Chat Completions nested format.

    LM Studio exposes an OpenAI-compatible ``chat.completions`` endpoint which
    requires the standard nested format used by the Chat Completions API:

        {"type": "function", "function": {"name": str, "description": str, "parameters": JSONSchema}}

    This is the *opposite* of the flat Responses-API format used by
    ``normalize_tool_schemas_for_openai``.

    Args:
        tools: Tool definitions in any supported format.

    Returns:
        List of tools in Chat Completions nested format.
    """
    normalized: list[JSONDict] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            # Already in nested Chat Completions format — pass through
            normalized.append(tool)
        elif tool.get("type") == "function" and "name" in tool:
            # Flat Responses-API format → wrap back into nested format
            nested: JSONDict = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            normalized.append(nested)
        else:
            normalized.append(tool)
    return normalized


def normalize_tool_schemas_for_anthropic(tools: list[JSONDict]) -> list[JSONDict]:
    """Normalize tool schemas to Anthropic tool format (same shape as OpenAI).

    Anthropic's tool format is very similar to OpenAI's, so this function
    primarily validates and passes through correctly formatted tools.

    Expected format (same as OpenAI):
        {
            "type": "function",
            "function": {
                "name": str,
                "description": str,
                "parameters": JSONSchema
            }
        }

    Args:
        tools: List of tool definitions to normalize

    Returns:
        List of tools in Anthropic-compatible format

    Note:
        Anthropic's tool handling is quite robust and accepts the same
        format as OpenAI, making this function primarily a pass-through
        with validation.
    """
    normalized: list[JSONDict] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            # OpenAI format → Anthropic custom tool format
            fn = tool["function"]
            anth_tool: JSONDict = {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
            normalized.append(anth_tool)
        elif "name" in tool and "input_schema" in tool:
            # Already in Anthropic format
            normalized.append(tool)
        else:
            # Best-effort pass-through
            normalized.append(tool)
    return normalized


def normalize_tool_schemas_for_gemini(
    tools: list[JSONDict],
) -> tuple[list[JSONDict], list[JSONDict]]:
    """Normalize tool schemas to Gemini function declarations format.

    Gemini uses a different tool format than OpenAI/Anthropic. Instead of
    wrapping functions in a "tools" array with type indicators, Gemini
    expects a flat list of function_declarations that get wrapped in a
    Tool object by the calling code.

    Conversion:
        OpenAI/Anthropic: [{"type": "function", "function": {...}}]
        Gemini: [{"name": ..., "description": ..., "parameters": ...}]

    Args:
        tools: List of tools in standard format

    Returns:
        Tuple of (function_declarations, other_tools):
        - function_declarations: Converted tools, to be wrapped in
          types.Tool(function_declarations=...) by the caller
        - other_tools: Tools passed through unchanged (e.g. tools already in
          native Gemini form), to be forwarded alongside the Tool object

    Note:
        The first element is used to create a types.Tool(function_declarations=...)
        object in the Gemini API implementation. This flattened format is specific
        to Google's API requirements.
    """
    function_declarations: list[JSONDict] = []
    other_tools: list[JSONDict] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            # Extract function definition from standard format
            fn = tool["function"]
            function_declarations.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                }
            )
        else:
            # If the tool already looks like a Gemini declaration, pass through
            other_tools.append(tool)

    return function_declarations, other_tools


def finalize_parsed_output(
    *,
    output_format: str,
    content: str,
) -> tuple[Any | None, JSONDict | None]:
    """Parse content into requested format and return (parsed, reasoning_meta).

    This function handles the final step of structured output processing,
    attempting to parse the raw text content based on the requested format.
    It uses safe parsing functions that never raise exceptions.

    Args:
        output_format: The requested output format ("text", "json", "yaml")
        content: The raw text content to parse
        json_schema: Optional JSON schema for validation (currently unused)

    Returns:
        Tuple of (parsed_data, reasoning_metadata):
        - parsed_data: Parsed object if successful, None if parsing failed or format was "text"
        - reasoning_metadata: Currently always None, reserved for future reasoning info

    Note:
        This function is designed to be safe and never raises exceptions.
        Parse failures result in None being returned for the parsed data,
        while the original content remains available in ChatResult.content.

        Future versions may use json_schema for validation when provided.
    """
    reasoning_meta: JSONDict | None = None

    if output_format == "json":
        parsed, _err = _try_parse_json(content)
        return parsed, reasoning_meta
    if output_format == "yaml":
        parsed, _err = _try_parse_yaml(content)
        return parsed, reasoning_meta

    # For "text" format or any unknown format, return None
    return None, reasoning_meta


def _safe_json_dumps(value: Any) -> str:
    """Safely convert a value to JSON string, falling back to str() on error.

    This utility function is used when we need to serialize values to JSON
    but cannot guarantee the value is JSON-serializable. It provides a
    safe fallback to string representation.

    Args:
        value: Any value to convert to JSON string

    Returns:
        JSON string representation of the value, or str(value) if
        JSON serialization fails

    Note:
        This function never raises exceptions. It's used primarily for
        tool result serialization where the return value type is unknown.
    """
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _normalize_tool_arguments(args: Any) -> tuple[dict[str, Any] | None, Any | None]:
    """Normalize tool call arguments for function execution.

    Different providers may pass tool arguments in various formats. This
    function normalizes them to either a dictionary of keyword arguments
    or a raw value for functions that accept single arguments.

    Args:
        args: Tool arguments in various formats (dict, JSON string, or raw value)

    Returns:
        Tuple of (kwargs_dict_or_none, raw_value_if_not_mapping):
        - kwargs_dict: Dictionary of keyword arguments if args is/parses to a dict
        - raw_value: Original value if it's not a mapping or parseable as one

    Processing Logic:
        - None -> ({}, None) - empty kwargs
        - Dict -> (dict, None) - direct keyword arguments
        - JSON string -> (parsed_dict, None) or (None, string) if not parseable as dict
        - Other values -> (None, value) - pass as single argument

    Note:
        This enables flexible function calling where tools can receive either
        structured arguments as **kwargs or simple values as single parameters.
    """
    if args is None:
        return {}, None
    if isinstance(args, dict):
        return args, None
    if isinstance(args, str):
        # Try to parse as JSON first
        parsed, _err = _try_parse_json(args)
        if isinstance(parsed, dict):
            return parsed, None
        # fallthrough to treat as raw string
        return None, args
    # non-mapping, non-string - pass as raw value
    return None, args


def execute_registered_functions(
    *,
    function_registry: dict[str, Callable[..., Any]],
    tool_calls: list[JSONDict] | None,
) -> tuple[list[JSONDict], list[JSONDict]]:
    """Execute registered functions for the given tool calls.

    This is the core function execution engine that takes tool calls from
    the LLM and executes the corresponding registered Python functions.
    It handles argument normalization, error capture, and result formatting.

    Args:
        function_registry: Dictionary mapping function names to callable objects
        tool_calls: List of tool call objects from the LLM response

    Returns:
        Tuple of (enriched_tool_calls_with_results, tool_result_messages):

        enriched_tool_calls: Original tool calls enhanced with execution results.
            Each call will include either a "result" key (on success) or an
            "error" key (on failure) in addition to the original call data.

        tool_result_messages: Provider-agnostic chat messages ready to be sent
            back to the LLM, with format:
            {"role": "tool", "name": <function_name>, "tool_call_id": <id>, "content": <result>}

    Function Execution Logic:
        1. Extract function name and arguments from each tool call
        2. Look up function in registry (skip if not found)
        3. Normalize arguments using _normalize_tool_arguments
        4. Execute function with appropriate calling convention:
           - func(**kwargs) if arguments are a dictionary
           - func(raw_value) if arguments are a single value
           - func() if no arguments
        5. Capture results or errors and format for LLM consumption

    Error Handling:
        - Missing functions are silently skipped (not an error)
        - Function execution errors are captured and included in results
        - No exceptions are raised; all errors are returned in the results

    Note:
        This function enables the LLM to call Python functions automatically,
        creating powerful integrations with external systems, databases, APIs, etc.
    """
    if not tool_calls:
        return [], []

    enriched_calls: list[JSONDict] = []
    tool_messages: list[JSONDict] = []

    for call in tool_calls:
        func_info = call.get("function", {}) if isinstance(call, dict) else {}
        func_name = func_info.get("name") if isinstance(func_info, dict) else None
        func_args = func_info.get("arguments") if isinstance(func_info, dict) else None
        call_id = call.get("id") if isinstance(call, dict) else None

        enriched: JSONDict = dict(call) if isinstance(call, dict) else {"raw": call}

        if not func_name or func_name not in function_registry:
            enriched_calls.append(enriched)
            tool_messages.append(
                {
                    "role": "tool",
                    "name": func_name,
                    "tool_call_id": call_id,
                    "content": {"error": f"Function: {func_name} not found"},
                }
            )
            continue

        kwargs, raw_value = _normalize_tool_arguments(func_args)
        try:
            if kwargs is not None:
                result_value = function_registry[func_name](**kwargs)
            elif raw_value is not None:
                result_value = function_registry[func_name](raw_value)
            else:
                result_value = function_registry[func_name]()
            enriched["result"] = result_value
            tool_messages.append(
                {
                    "role": "tool",
                    "name": func_name,
                    "tool_call_id": call_id,
                    "content": result_value,
                }
            )
        except Exception as exc:  # noqa: BLE001 - surface tool exec errors
            err_text = f"{type(exc).__name__}: {exc}"
            enriched["error"] = err_text
            tool_messages.append(
                {
                    "role": "tool",
                    "name": func_name,
                    "tool_call_id": call_id,
                    "content": {"error": err_text},
                }
            )

        enriched_calls.append(enriched)

    return enriched_calls, tool_messages
