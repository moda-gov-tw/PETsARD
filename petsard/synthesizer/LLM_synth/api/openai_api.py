"""OpenAI API implementation built on APITemplate.

This module provides a complete OpenAI API integration that supports all major
features while maintaining compatibility with the unified API interface.

Key Features:
- Full OpenAI Responses API support with automatic function execution
- Native structured output via response_format (json_object/json_schema)
- YAML output via prompting and post-processing
- Chain-of-thought reasoning for O3/O4 and other reasoning models
- Comprehensive tool/function calling with automatic result handling
- Token usage tracking and cost optimization

OpenAI-Specific Features:
- Responses API for improved function calling and reasoning
- Native JSON schema validation for structured outputs
- Reasoning effort control for O3/O4 models
- Advanced tool choice options ("auto", "none", specific tools)
- Context length optimization with intelligent message handling

Supported Models:
- GPT-4o family (gpt-4o, gpt-4o-mini)
- O3/O4 reasoning models (when available)
- Legacy GPT-3.5 and GPT-4 models
- Custom fine-tuned models

Authentication:
Requires a valid OpenAI API key. The key should have access to the models
you intend to use and sufficient credits for your usage patterns.
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    APITemplate,
    ChatResult,
    ChatSession,
    execute_registered_functions,
    finalize_parsed_output,
    normalize_tool_schemas_for_openai,
    raise_for_response_error,
)


class OpenAIAPI(APITemplate):
    """OpenAI API implementation with full Responses API support.

    This class provides a complete integration with OpenAI's API, including
    support for the latest Responses API which offers improved function calling,
    reasoning capabilities, and structured outputs. It automatically handles
    authentication, request formatting, response parsing, and tool execution.

    Notable Implementation Details:
    - Uses OpenAI's Responses API for enhanced capabilities
    - Automatically detects and enables reasoning for O3/O4 models
    - Handles tool execution flow with automatic follow-up requests
    - Provides comprehensive error handling and usage tracking
    """

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Initialize OpenAI API client.

        Args:
            api_key: Your OpenAI API key (starts with 'sk-')
            model: Model to use. Defaults to "gpt-4o-mini" for cost efficiency.
                  Common options: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
        """
        super().__init__(api_key=api_key, model=model or "gpt-4o-mini")
        self._client = None
        self._ensure_client()

    def _ensure_client(self) -> None:
        """Initialize the OpenAI client, handling import and authentication errors gracefully.

        This method uses lazy initialization to avoid import errors when the OpenAI
        library is not installed. Errors are stored and raised later when _send_messages
        is called, providing better error messages and allowing the class to be
        imported even without the OpenAI library.
        """
        if self._client is not None:
            return
        try:
            # OpenAI Python SDK v1 style
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001 - surface import errors at call time
            self._client = exc  # store error to raise later

    def _supports_cot(self) -> bool:
        """Detect if the current model supports chain-of-thought reasoning.

        Returns:
            True if the model supports reasoning capabilities, False otherwise.

        Note:
            This uses heuristic model name detection. OpenAI's reasoning models
            typically include "o3", "o4", "reasoning", or similar identifiers.
        """
        name = (self._model or "").lower()
        # Heuristic: reasoning lines typically include o3/o4 naming
        return any(tag in name for tag in ["o3", "o4", "reasoning", "gpt-5"])

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
        """Create a ChatSession bound to this OpenAIAPI instance.

        Automatically enables chain-of-thought reasoning for supported models
        and configures OpenAI-specific optimizations.

        Args:
            tools: Tool definitions for function calling
            output_format: Response format ("text", "json", "yaml")
            json_schema: JSON schema for structured output validation
            enable_cot: Whether to enable reasoning for supported models
            session_params: Unified parameters (translated before send), e.g.:
                - max_tokens: Output cap (→ max_output_tokens on Responses API)
                - reasoning_effort: "low", "medium", "high" for O3/O4 models
                - temperature: Sampling temperature (0.0-2.0)
                - top_p: Nucleus sampling parameter
            custom_messages: Pre-existing conversation history

        Returns:
            ChatSession configured for OpenAI API with automatic reasoning detection
        """
        return super().create_chat_session(
            tools=tools,
            output_format=output_format,
            json_schema=json_schema,
            enable_cot=self._supports_cot() if enable_cot is None else enable_cot,
            session_params=session_params,
            custom_messages=custom_messages,
        )

    def _prepare_user_message(self, user_message: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": user_message}] if user_message else []

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
        """Send messages to OpenAI API using the Responses API.

        This method handles the complete request flow including:
        1. Message format conversion for Responses API
        2. Tool/function call setup and execution
        3. Structured output configuration
        4. Reasoning model configuration for O3/O4
        5. Automatic tool result handling with follow-up requests
        6. Response parsing and normalization

        The implementation uses OpenAI's Responses API which provides enhanced
        capabilities compared to the standard chat completions API, including
        better function calling and reasoning support.

        Args:
            messages: Conversation history in standard format
            tools: Function definitions for tool calling
            tool_choice: Tool selection strategy or specific tool
            output_format: Desired response format
            json_schema: Schema for structured output validation
            enable_cot: Whether to enable reasoning for supported models
            request_params: Request-specific parameters
            session_params: Session-level parameters

        Returns:
            ChatResult with normalized response data

        Raises:
            Exception: Re-raises any API errors from OpenAI
        """
        if isinstance(self._client, Exception):
            raise self._client
        client = self._client

        # Prepare tool configuration
        oai_tools = normalize_tool_schemas_for_openai(tools or []) if tools else None

        # Prepare response_format
        response_format = None
        if output_format == "json":
            if json_schema:
                # Newer SDK supports json_schema in chat.completions
                response_format = {"type": "json_schema"} | json_schema
            else:
                response_format = {"type": "json_object"}
        output = []

        api_params = self._translate_request_params(
            {**(session_params or {}), **(request_params or {})}
        )
        kwargs_params = {
            k: v for k, v in api_params.items() if k != "reasoning_effort"
        }
        default_effort = api_params.get("reasoning_effort")

        # Perform request
        try:
            # Build Responses API payload
            # Convert chat messages to Responses `input` prompt messages
            input_messages: list[dict[str, Any]] = []
            for msg in messages:
                if type(msg) is dict:
                    role_val = msg.get("role")
                    if role_val == "system":
                        role_val = "developer"
                    content_val = msg.get("content", "")
                    if type(content_val) is str:
                        content_val = [
                            {
                            "type": "input_text" if role_val != "assistant" else "output_text",
                            "text": content_val
                            }
                        ]
                    base_msg: dict[str, Any] = {
                        "role": role_val,
                        "content": content_val,
                    }
                    # If this is a tool result message, forward tool_call_id if present
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
            # Enable reasoning for reasoning-capable models
            if enable_cot:
                kwargs["reasoning"] = {
                    "effort": default_effort or "medium",
                    #"summary": "auto"
                    }
                kwargs["include"] = [ "reasoning.encrypted_content" ]
            if oai_tools:
                kwargs["tools"] = oai_tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            if response_format:
                kwargs["text"] = {"format" :response_format}
            # Pass through additional params (session + request, translated)
            kwargs.update(kwargs_params)

            #if response_format:
            #    response = client.responses.parse(**kwargs)
            #else:
            response = client.responses.create(**kwargs)  # type: ignore[attr-defined]
            # The API may return normally but report a failure in response.error
            raise_for_response_error(response, provider="OpenAI", model=self._model)
            output += response.output

            content = ""
            tool_calls: list[dict[str, Any]] = []
            thinking_blocks: list[str] = []
            usage_list: list[dict[str, Any]] = []
            enriched_tool_calls = None
            tool_calls_for_return: list[dict[str, Any]] = []
            parsed = None
            reasoning_meta = None

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
                    cached_tokens = 0
                    input_tokens_details = getattr(usage_obj, "input_tokens_details", None)
                    if input_tokens_details:
                        cached_tokens = getattr(input_tokens_details, "cached_tokens", 0) or 0
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
                try:
                    choice_reasoning = getattr(response, "reasoning", None)
                    if choice_reasoning:
                        if isinstance(choice_reasoning, list):
                            thinking_blocks.extend([str(x) for x in choice_reasoning])
                        else:
                            thinking_blocks.append(str(choice_reasoning))
                except Exception:
                    pass

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
                        loop_kwargs["include"] = ["reasoning.encrypted_content"]
                    if oai_tools:
                        loop_kwargs["tools"] = oai_tools
                    if response_format:
                        loop_kwargs["text"] = {"format": response_format}
                    loop_kwargs.update(kwargs_params)

                    response = client.responses.create(**loop_kwargs)  # type: ignore[attr-defined]
                    raise_for_response_error(response, provider="OpenAI", model=self._model)
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

            if thinking_blocks:
                if reasoning_meta is None:
                    reasoning_meta = {}
                reasoning_meta["thinking_blocks"] = thinking_blocks

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
                reasoning=reasoning_meta,
                output=output,
                id=response.id,
            )
        except Exception as exc:  # noqa: BLE001 - surface provider errors
            raise exc


    def _translate_request_params(
        self,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Map unified params to OpenAI Responses API (``responses.create``).

        - ``max_tokens`` → ``max_output_tokens`` (Responses API name)
        - ``temperature``, ``top_p``: forwarded as-is
        - ``reasoning_effort``: kept for ``_send_messages`` (not passed in kwargs)
        """
        params = dict(request_params)
        if "max_tokens" in params and "max_output_tokens" not in params:
            params["max_output_tokens"] = params.pop("max_tokens")
        elif "max_tokens" in params:
            del params["max_tokens"]
        return params
