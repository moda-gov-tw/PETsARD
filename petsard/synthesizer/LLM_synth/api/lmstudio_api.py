"""LM Studio API implementation built on APITemplate.

This module provides a comprehensive integration with LM Studio's local server,
offering OpenAI-compatible endpoints while maintaining compatibility with the
unified API interface.

Key Features:
- Local LLM execution through LM Studio's server
- OpenAI Responses API support (client.responses.create)
- Native structured output via text.format (json_object/json_schema)
- Tool/function calling with automatic execution
- Support for both GGUF and MLX models
- No API key required for local operation
- Enhanced statistics (tokens/second, time to first token)

LM Studio-Specific Features:
- Local server operation (default: http://localhost:1234/v1)
- Grammar-based sampling for GGUF models (via llama.cpp)
- Outlines-based sampling for MLX models
- Enhanced model information and runtime stats
- Support for custom server URLs and ports
- Tool use with native and default support

Supported Models:
- Any GGUF model loaded in LM Studio (via llama.cpp)
- Any MLX model loaded in LM Studio (Apple Silicon only)
- Models with native tool use support (Qwen, Llama-3.x, Mistral)
- Models with default tool use support (all others)

Authentication:
No API key required for a local server. Requires LM Studio to be running with
the server started. Default endpoint is http://localhost:1234/v1 but can be
customized via the api_url parameter. If api_url is not set, the api_key
parameter is used as the base_url (backward compatible). The api_key value is
always sent to the server as the API key. A plain-http URL whose host is not
this machine (localhost / 127.0.0.1 / ::1) triggers a warning, since prompts
would travel unencrypted.

Note on Local Operation:
This implementation connects to a locally running LM Studio instance.
Ensure LM Studio is running and the server is started before making API calls.
You can start the server via: `lms server start` or through the LM Studio UI.
"""

from __future__ import annotations

import ipaddress
import json
import warnings
from typing import Any
from urllib.parse import urlsplit

from .base import (
    APIResponseError,
    APITemplate,
    ChatResult,
    ChatSession,
    _get_field,
    execute_registered_functions,
    finalize_parsed_output,
    normalize_tool_schemas_for_openai,
    raise_for_response_error,
)


def _warn_if_remote_plain_http(base_url: str) -> None:
    """Warn when requests would go unencrypted (http) to a host other than this machine."""
    parts = urlsplit(base_url)
    if parts.scheme.lower() != "http":
        return
    host = (parts.hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    warnings.warn(
        f"LM Studio server '{host}' is reached over plain http: prompts, which can contain "
        "training records, leave this machine unencrypted. Use a local server "
        "(localhost / 127.0.0.1 / ::1) or an https URL.",
        stacklevel=3,
    )


class LMStudioAPI(APITemplate):
    """LM Studio API implementation with local server support.

    This class provides a complete integration with LM Studio's local server,
    including support for structured outputs, tool calling, and all OpenAI-compatible
    features. It automatically handles local server communication and provides
    enhanced statistics specific to LM Studio.

    Key Implementation Features:
    - Connects to local LM Studio server (no cloud dependency)
    - Uses OpenAI Responses API (client.responses.create)
    - Automatic tool execution with follow-up conversations
    - Enhanced statistics (tokens/second, TTFT, generation time)
    - Support for both GGUF and MLX model formats
    - Flexible server URL configuration
    """

    def __init__(
        self,
        api_key: str = "http://localhost:1234/v1",
        model: str | None = None,
        api_url: str | None = None,
    ) -> None:
        """Initialize LM Studio API client.

        Args:
            api_key: The API key sent to the server. When api_url is not set,
                    this value is also used as the base URL of the LM Studio
                    server (kept for backward compatibility).
                    Defaults to "http://localhost:1234/v1" for local server.
            model: Model identifier to use. This should match a model loaded in
                  LM Studio. If None, you must specify the model when creating
                  a chat session or LM Studio will use the currently loaded model.
                  Examples:
                  - "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
                  - "lmstudio-community/Qwen2.5-7B-Instruct-GGUF"
                  - "mlx-community/Qwen2.5-7B-Instruct-4bit"
            api_url: The base URL for the LM Studio server. If None, api_key
                    is used as the base URL instead.
                    Examples:
                    - "http://localhost:1234/v1" (default local)
                    - "http://192.168.1.100:1234/v1" (network server)
                    - "http://localhost:5000/v1" (custom port)

        Note:
            A local LM Studio server doesn't require authentication, so api_key
            can be any value (including the server URL). When the server sits
            behind an authenticated endpoint, pass the real key as api_key and
            the server URL as api_url.

            LM Studio server must be running before making API calls.
            Start it with: `lms server start` or through the LM Studio UI.
        """
        # Use api_url as base_url when provided; otherwise fall back to api_key
        url = api_url if api_url else api_key
        self._base_url = url if url.startswith("http") else f"http://{url}"
        if not self._base_url.endswith("/v1"):
            self._base_url = f"{self._base_url}/v1"
        _warn_if_remote_plain_http(self._base_url)

        super().__init__(api_key=api_key, model=model, api_url=self._base_url)
        self._client = None
        self._model_info_cache: dict[str, Any] | None = None  # Cache for model info
        self._ensure_client()

    def _ensure_client(self) -> None:
        """Initialize the OpenAI client pointing to LM Studio's local server.

        This method uses lazy initialization to avoid import errors when the OpenAI
        library is not installed. The client is configured to point to the local
        LM Studio server instead of OpenAI's cloud servers.

        Any initialization errors are stored and raised later during actual API calls,
        providing better error messages and allowing the class to be imported even
        without the OpenAI library.
        """
        if self._client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore

            # Configure client to point to local LM Studio server
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        except Exception as exc:  # noqa: BLE001 - surface import errors at call time
            self._client = exc  # store error to raise later

    def _get_model_info(self) -> dict[str, Any] | None:
        """Query LM Studio API for model information.

        Returns:
            Dictionary containing model information from LM Studio API, or None if query fails.

        Note:
            This method caches the result to avoid repeated API calls.
            Model info includes: arch, type, compatibility_type, quantization, etc.
        """
        # Return cached info if available
        if self._model_info_cache is not None:
            return self._model_info_cache

        if not self._model:
            return None

        try:
            import requests  # type: ignore

            # Try LM Studio's native API first for more detailed info
            # Remove /v1 suffix and use /api/v0/models endpoint
            base_url_without_v1 = self._base_url.replace("/v1", "")

            # Try to get specific model info
            response = requests.get(
                f"{base_url_without_v1}/api/v0/models/{self._model}",
                timeout=2
            )

            if response.status_code == 200:
                self._model_info_cache = response.json()
                return self._model_info_cache

            # Fallback: Try to list all models and find ours
            response = requests.get(
                f"{base_url_without_v1}/api/v0/models",
                timeout=2
            )

            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data:
                    for model in models_data["data"]:
                        if model.get("id") == self._model:
                            self._model_info_cache = model
                            return self._model_info_cache

        except Exception:
            # If requests library not available or API call fails, return None
            pass

        return None

    def _responses_create_params(self) -> set:
        """Keyword names client.responses.create takes."""
        import inspect

        return set(inspect.signature(self._client.responses.create).parameters)

    def _create_response(self, **kwargs: Any) -> Any:
        """Call client.responses.create as a stream and return its final response.

        A streamed request has data on the connection for as long as the model
        works, so a proxy that drops idle connections (some reverse proxies do
        after about 100 seconds) does not cut a long reply short. The response
        returned is the object a non-streamed request returns: the one the
        stream's last event carries.

        Args:
            **kwargs: Arguments of client.responses.create; ``stream`` is set here.

        Returns:
            The response, whatever its status ("completed", "incomplete", "failed").

        Raises:
            APIResponseError: If the stream reports an error event.
            ConnectionError: If the stream ends without its final response.
        """
        response = None
        with self._client.responses.create(**{**kwargs, "stream": True}) as stream:  # type: ignore[attr-defined]
            for event in stream:
                kind = _get_field(event, "type")
                if kind in ("response.completed", "response.incomplete", "response.failed"):
                    response = _get_field(event, "response")
                elif kind == "error":
                    raise APIResponseError(
                        str(_get_field(event, "message") or "the stream reported an error"),
                        provider="LMStudio",
                        code=_get_field(event, "code"),
                        model=self._model,
                    )
        if response is None:
            raise ConnectionError("the connection closed before the response was complete")
        return response

    def _supports_cot(self) -> bool:
        name = (self._model or "").lower()
        # LM Studio reasoning models often include "oss" in the name
        return "oss" in name

    def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of available models from LM Studio server.

        Returns:
            List of dictionaries containing model information. Each dictionary
            includes fields like: id, arch, type, quantization, state, etc.
            Returns empty list if query fails.

        Example:
            api = LMStudioAPI()
            models = api.get_available_models()
            for model in models:
                print(f"{model['id']}: {model.get('arch')} ({model.get('state')})")
        """
        try:
            import requests  # type: ignore

            # Use LM Studio's native API for detailed model info
            base_url_without_v1 = self._base_url.replace("/v1", "")
            response = requests.get(
                f"{base_url_without_v1}/api/v0/models",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
        except Exception:
            pass

        # Fallback to OpenAI-compatible endpoint
        try:
            if isinstance(self._client, Exception):
                return []
            client = self._client

            models_response = client.models.list()
            return [
                {"id": model.id, "object": model.object}
                for model in getattr(models_response, "data", [])
            ]
        except Exception:
            return []

    def create_chat_session(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        output_format: str = "text",
        json_schema: dict[str, Any] | None = None,
        enable_cot: bool | None = None,
        session_params: dict[str, Any] | None = None,
        custom_messages: list[dict[str, Any]] | None = None,
    ) -> ChatSession:
        """Create a ChatSession bound to this LMStudioAPI instance.

        Configures LM Studio-specific optimizations for local model execution,
        structured outputs, and tool calling.

        Args:
            tools: Tool definitions for function calling. These will be normalized
                  to OpenAI format which is compatible with LM Studio.
                  LM Studio supports both native and default tool use modes.
            output_format: Response format ("text", "json", "yaml").
                         LM Studio has native JSON schema support via response_format.
                         YAML is achieved through prompting and post-processing.
            json_schema: JSON schema for structured output validation.
                        LM Studio uses grammar-based sampling (GGUF models via llama.cpp)
                        or Outlines (MLX models) to enforce schema compliance.
            enable_cot: Whether to enable reasoning for supported models
            session_params: LM Studio-specific parameters like:
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum response length (-1 for unlimited)
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - repeat_penalty: Repetition penalty
                - reasoning_effort: Reasoning effort for models that support it
                  ("low", "medium", "high"). Default: "medium"
            custom_messages: Pre-existing conversation history

        Returns:
            ChatSession configured for LM Studio API with local model execution
            and enhanced statistics.

        Note:
            LM Studio provides enhanced statistics including tokens/second and
            time to first token (TTFT) which can be useful for performance
            monitoring and optimization.

            For reasoning-capable models (e.g., gpt-oss-20b), set reasoning_effort
            in session_params to control reasoning depth.
        """
        return super().create_chat_session(
            tools=tools,
            output_format=output_format,
            json_schema=json_schema,
            enable_cot=self._supports_cot() if enable_cot is None else enable_cot,
            session_params=session_params,
            custom_messages=custom_messages,
        )

    # ---------------------- provider-specific send ---------------------- #
    def _send_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        output_format: str,
        json_schema: dict[str, Any] | None,
        enable_cot: bool,
        request_params: dict[str, Any],
        session_params: dict[str, Any],
    ) -> ChatResult:
        """Send messages to LM Studio's local server using the Responses API.

        This method handles the complete request flow including:
        1. Message format conversion for Responses API input format
        2. Tool/function call setup and execution
        3. Structured output configuration via text.format
        4. Request to local LM Studio server via client.responses.create
        5. Automatic tool result handling with follow-up requests
        6. Response parsing with enhanced LM Studio statistics
        7. Usage tracking and error handling

        LM Studio-Specific Features:
        - Local server communication (no cloud dependency)
        - Enhanced statistics: tokens/second, TTFT, generation_time
        - Grammar-based sampling for structured output (GGUF)
        - Outlines-based sampling for structured output (MLX)
        - Native and default tool use support
        - Model information in response (arch, quant, format)
        - <think> tag extraction from model output

        Args:
            messages: Conversation history in standard format
            tools: Function definitions for tool calling (OpenAI format)
            tool_choice: Tool selection strategy or specific tool
            output_format: Desired response format ("text", "json", "yaml")
            json_schema: Schema for structured output validation
            enable_cot: Whether to enable reasoning (model-dependent)
            request_params: Request-specific parameters (temperature, max_tokens, etc.)
            session_params: Session-level parameters that persist across requests

        Returns:
            ChatResult with normalized response data including LM Studio's
            enhanced statistics and model information

        Raises:
            Exception: Re-raises any API errors from LM Studio server,
                      including connection errors if server is not running

        Note:
            Ensure LM Studio server is running before calling this method.
            Connection errors typically indicate the server is not started.
            Start server with: `lms server start` or through LM Studio UI.
        """
        if isinstance(self._client, Exception):
            raise self._client

        # Prepare tool configuration (OpenAI format is compatible with LM Studio)
        lms_tools = normalize_tool_schemas_for_openai(tools or []) if tools else None

        # Prepare response_format for structured output (Responses API uses text.format)
        response_format = None
        if output_format == "json":
            if json_schema:
                response_format = {"type": "json_schema"} | json_schema
            else:
                response_format = {"type": "json_object"}

        output = []

        api_params = self._translate_request_params(
            {**(session_params or {}), **(request_params or {})}
        )
        accepted = self._responses_create_params()
        kwargs_params = {
            k: v for k, v in api_params.items()
            if k != "reasoning_effort" and k in accepted
        }
        extra_body = {
            k: v for k, v in api_params.items()
            if k != "reasoning_effort" and k not in accepted
        }
        default_effort = api_params.get("reasoning_effort")

        # Perform request to local LM Studio server
        try:
            # Convert chat messages to Responses API `input` format
            input_messages: list[dict[str, Any]] = []
            for msg in messages:
                if type(msg) is dict:
                    role_val = msg.get("role")
                    base_msg: dict[str, Any] = {
                        "role": role_val,
                        "content": msg.get("content", ""),
                    }
                    if role_val == "tool":
                        base_msg = {
                            "type": "function_call_output",
                            "call_id": msg.get("tool_call_id"),
                            "output": msg.get("content"),
                        }
                    input_messages.append(base_msg)
                else:
                    input_messages.append(msg)

            kwargs: dict[str, Any] = {
                "model": self._model,
                "input": input_messages,
            }

            # Add reasoning support for models that support it (gpt-oss-20b, etc.)
            if enable_cot:
                kwargs["reasoning"] = {"effort": default_effort or "medium"}

            if lms_tools:
                kwargs["tools"] = lms_tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            if response_format:
                kwargs["text"] = {"format": response_format}
            if "timeout" not in kwargs:
                kwargs["timeout"] = 60*60*3 # 3 hours

            # Session + request parameters (temperature, max_output_tokens, etc.)
            kwargs.update(kwargs_params)
            if extra_body:
                kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **extra_body}

            response = self._create_response(**kwargs)
            # The server may return normally but report a failure in response.error
            raise_for_response_error(response, provider="LMStudio", model=self._model)
            output += response.output

            content = ""
            tool_calls: list[dict[str, Any]] = []
            thinking_blocks: list[str] = []
            usage_list: list[dict[str, Any]] = []
            enriched_tool_calls = None
            tool_calls_for_return: list[dict[str, Any]] = []
            parsed = None
            reasoning_meta: dict[str, Any] | None = None

            while True:
                # Collect usage from current response
                if getattr(response, "usage", None) is not None:
                    usage_obj = response.usage
                    prompt_tokens = getattr(usage_obj, "prompt_tokens", None) or getattr(usage_obj, "input_tokens", None) or 0
                    completion_tokens = getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", None) or 0
                    total_tokens = getattr(usage_obj, "total_tokens", None)
                    if total_tokens is None:
                        try:
                            total_tokens = int(prompt_tokens) + int(completion_tokens)
                        except Exception:
                            total_tokens = None
                    cached_tokens = _get_field(
                        _get_field(usage_obj, "input_tokens_details"), "cached_tokens"
                    ) or 0
                    usage_list.append({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens or 0,
                        "input_tokens": prompt_tokens - cached_tokens,
                        "cached_tokens": cached_tokens,
                    })

                # Prefer Responses API output_text
                content = getattr(response, "output_text", None) or ""
                parsed, reasoning_meta = finalize_parsed_output(
                    output_format=output_format, content=content
                )

                # Extract tool calls from response.output
                tool_calls = []
                try:
                    for out_item in getattr(response, "output", []) or []:
                        otype = getattr(out_item, "type", None) or (
                            out_item.get("type") if isinstance(out_item, dict) else None
                        )
                        if otype in {"tool_call", "function_call"}:
                            call_id = getattr(out_item, "call_id", None) or (
                                out_item.get("call_id") if isinstance(out_item, dict) else None
                            )
                            name = (
                                getattr(out_item, "name", None)
                                or (out_item.get("name") if isinstance(out_item, dict) else None)
                            )
                            arguments = (
                                getattr(out_item, "arguments", None)
                                or (out_item.get("arguments") if isinstance(out_item, dict) else None)
                            )
                            if name is not None:
                                tool_calls.append({
                                    "type": "function",
                                    "id": call_id,
                                    "function": {"name": name, "arguments": arguments},
                                })
                except Exception:
                    pass

                # Collect reasoning from this response
                for out_item in getattr(response, "output", []) or []:
                    if _get_field(out_item, "type") != "reasoning":
                        continue
                    parts = list(_get_field(out_item, "content") or []) + list(
                        _get_field(out_item, "summary") or []
                    )
                    text = "\n".join(_get_field(p, "text") or "" for p in parts).strip()
                    if text:
                        thinking_blocks.append(text)

                if not tool_calls:
                    break

                tool_calls_for_return = tool_calls
                enriched_tool_calls, tool_result_messages = execute_registered_functions(
                    function_registry=self._function_registry,
                    tool_calls=tool_calls,
                )

                if not tool_result_messages:
                    break

                try:
                    function_call_results = []
                    for tr in tool_result_messages:
                        function_call_results.append({
                            "type": "function_call_output",
                            "call_id": tr.get("tool_call_id"),
                            "output": json.dumps({tr.get("name"): tr.get("content", "")}),
                        })
                    loop_kwargs: dict[str, Any] = {
                        "model": self._model,
                        "input": input_messages + output + function_call_results,
                    }
                    if enable_cot:
                        loop_kwargs["reasoning"] = {"effort": default_effort or "medium"}
                    if lms_tools:
                        loop_kwargs["tools"] = lms_tools
                    if response_format:
                        loop_kwargs["text"] = {"format": response_format}
                    if "timeout" not in loop_kwargs:
                        loop_kwargs["timeout"] = 60*60*3 # 3 hours
                    loop_kwargs.update(kwargs_params)
                    if extra_body:
                        loop_kwargs["extra_body"] = {**(loop_kwargs.get("extra_body") or {}),
                                                     **extra_body}

                    response = self._create_response(**loop_kwargs)
                    raise_for_response_error(response, provider="LMStudio", model=self._model)
                    output += function_call_results + response.output
                except Exception as exc:  # noqa: BLE001
                    raise exc

            # Aggregate usage across all turns
            usage = None
            usage_input_token = 0
            usage_cache_input_token = 0
            if usage_list:
                usage = {
                    "prompt_tokens": sum(u["prompt_tokens"] for u in usage_list),
                    "completion_tokens": sum(u["completion_tokens"] for u in usage_list),
                    "total_tokens": sum(u["total_tokens"] for u in usage_list),
                    "input_tokens": sum(u["input_tokens"] for u in usage_list),
                    "cached_tokens": sum(u["cached_tokens"] for u in usage_list),
                }
                usage_input_token = usage["input_tokens"]
                usage_cache_input_token = usage["cached_tokens"]

            # Build reasoning metadata
            if thinking_blocks:
                if reasoning_meta is None:
                    reasoning_meta = {}
                reasoning_meta["thinking_blocks"] = thinking_blocks

            if reasoning_meta is None:
                reasoning_meta = {}

            # Try to extract LM Studio-specific stats if available
            try:
                if hasattr(response, "stats"):
                    stats = response.stats
                    reasoning_meta["lmstudio_stats"] = {
                        "tokens_per_second": getattr(stats, "tokens_per_second", None),
                        "time_to_first_token": getattr(stats, "time_to_first_token", None),
                        "generation_time": getattr(stats, "generation_time", None),
                        "stop_reason": getattr(stats, "stop_reason", None),
                    }
                if hasattr(response, "model_info"):
                    model_info = response.model_info
                    reasoning_meta["model_info"] = {
                        "arch": getattr(model_info, "arch", None),
                        "quant": getattr(model_info, "quant", None),
                        "format": getattr(model_info, "format", None),
                        "context_length": getattr(model_info, "context_length", None),
                    }
            except Exception:
                pass  # Enhanced stats not available, continue without them

            # Check for <think> tags in content and extract them to reasoning
            if content and "<think>" in content and "</think>" in content:
                import re
                think_pattern = r"<think>(.*?)</think>"
                think_matches = re.findall(think_pattern, content, flags=re.DOTALL)

                if think_matches:
                    if reasoning_meta is None:
                        reasoning_meta = {}

                    current_blocks = reasoning_meta.get("thinking_blocks", [])
                    if not isinstance(current_blocks, list):
                        current_blocks = []

                    for block in think_matches:
                        current_blocks.append(block.strip())

                    reasoning_meta["thinking_blocks"] = current_blocks

                    # Remove <think> tags from content
                    content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()

            return ChatResult(
                content=content,
                parsed=parsed,
                format=output_format,
                tool_calls=enriched_tool_calls if enriched_tool_calls is not None else (tool_calls_for_return or None),
                usage=usage,
                usage_input_token=usage_input_token,
                usage_cache_input_token=usage_cache_input_token,
                usage_output_token=usage["completion_tokens"] if usage else 0,
                raw=response,
                reasoning=reasoning_meta if reasoning_meta else None,
                output=output,
                id=getattr(response, "id", "lmstudio-unknown"),
            )
        except APIResponseError:
            # Error reported by the server itself; not a connection issue
            raise
        except Exception as exc:  # noqa: BLE001 - surface provider errors
            # Provide helpful error message for common connection issues
            if "Connection" in str(exc) or "connection" in str(exc):
                raise RuntimeError(
                    f"Failed to connect to LM Studio server at {self._base_url}. "
                    "Please ensure LM Studio is running and the server is started. "
                    "Start server with: 'lms server start' or through LM Studio UI."
                ) from exc
            raise exc


    def _translate_request_params(
        self,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Map unified params to LM Studio Responses API (``responses.create``).

        Same renaming as OpenAIAPI: ``max_tokens`` → ``max_output_tokens``.
        ``temperature``, ``top_p``, and ``reasoning_effort`` are unchanged
        (``reasoning_effort`` is consumed in ``_send_messages``, not kwargs).
        """
        params = dict(request_params)
        if "max_tokens" in params and "max_output_tokens" not in params:
            params["max_output_tokens"] = params.pop("max_tokens")
        elif "max_tokens" in params:
            del params["max_tokens"]
        return params
