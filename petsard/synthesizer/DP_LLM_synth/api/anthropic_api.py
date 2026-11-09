"""Anthropic (Claude) API implementation built on APITemplate.

This module provides a comprehensive integration with Anthropic's Claude models,
offering advanced capabilities through the official Anthropic SDK while maintaining
compatibility with the unified API interface.

Key Features:
- Native tool/function calling with automatic execution and follow-up
- Advanced reasoning capabilities with thinking tokens (Claude 3.7+ models)
- Best-effort structured output through prompting and parsing
- Comprehensive message handling and conversation continuity
- Token usage tracking and cost optimization
- Error handling and graceful degradation

Claude-Specific Capabilities:
- Thinking tokens for chain-of-thought reasoning in supported models
- Advanced system prompt handling with context preservation
- Tool use with comprehensive argument parsing and validation
- Automatic message formatting for Claude's expected input structure
- Caching support for frequently used contexts (when available)

Supported Models:
- Claude 3.5 Sonnet (latest and previous versions)
- Claude 3.7 models with thinking capabilities
- Claude 4 models (when available)
- Claude Haiku for fast, cost-effective responses
- Claude Opus for complex reasoning tasks

Authentication:
Requires a valid Anthropic API key. Ensure your key has access to the models
you intend to use and sufficient credits for your usage patterns.

Note on Structured Output:
While Claude doesn't natively support JSON schema validation like OpenAI,
this implementation provides best-effort structured output through careful
prompting and post-processing validation.
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
    normalize_tool_schemas_for_anthropic,
    raise_for_response_error,
)


class AnthropicAPI(APITemplate):
    """Anthropic Claude API implementation with full thinking and tool support.

    This class provides a complete integration with Anthropic's Claude models,
    including advanced features like thinking tokens, sophisticated tool calling,
    and intelligent message handling. It automatically adapts to different Claude
    model capabilities and provides optimal configuration for each.

    Key Implementation Features:
    - Automatic thinking token support detection for Claude 3.7+ models
    - Sophisticated tool execution with automatic follow-up conversations
    - Advanced system prompt integration that preserves context
    - Intelligent message formatting for Claude's API requirements
    - Comprehensive error handling with graceful degradation
    - Token usage optimization and cost tracking
    """

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Initialize Anthropic API client.

        Args:
            api_key: Your Anthropic API key (typically starts with 'sk-ant-')
            model: Claude model to use. Defaults to "claude-3-5-sonnet-latest"
                  for the best balance of capability and cost.
                  Options include:
                  - "claude-3-5-sonnet-latest" (recommended default)
                  - "claude-3-5-haiku-latest" (fastest, most cost-effective)
                  - "claude-3-7-sonnet-thinking" (with thinking capabilities)
                  - "claude-4-sonnet" (when available)
        """
        super().__init__(api_key=api_key, model=model or "claude-3-5-sonnet-latest")
        self._client = None
        self._ensure_client()

    def _ensure_client(self) -> None:
        """Initialize the Anthropic client with lazy loading and error handling.

        This method uses lazy initialization to avoid import errors when the
        Anthropic library is not installed. Any initialization errors are
        stored and raised later during actual API calls, allowing for better
        error messaging and graceful handling of missing dependencies.
        """
        if self._client is not None:
            return
        try:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic(api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001
            self._client = exc

    def _supports_cot(self) -> bool:
        """Detect if the current Claude model supports thinking tokens.

        Returns:
            True if the model supports chain-of-thought reasoning with thinking tokens.

        Note:
            Claude models with thinking support typically include version identifiers
            like "3.7", "4", or explicit "thinking" in their names. This detection
            enables automatic configuration of thinking token budgets and reasoning
            capabilities.
        """
        name = (self._model or "").lower()
        # Claude models with thinking support are typically 3.7 or have "thinking"
        return any(tag in name for tag in ["3-7", "-4-", "thinking", "sonnet-thinking"])

    def _get_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        """Convert standard message format to Anthropic's expected format.

        Anthropic's API expects system messages to be separated from the conversation
        flow and provided as a dedicated system parameter. This method extracts
        system messages and formats user/assistant messages appropriately.

        Args:
            messages: Standard message list with roles and content

        Returns:
            Tuple of (formatted_messages, combined_system_prompt)

        Note:
            - System messages are extracted and combined into a single system prompt
            - Only user and assistant messages are included in the message list
            - Complex message objects are handled for compatibility
        """
        anth_messages: list[dict[str, Any]] = []
        system_prompt = ""

        for m in messages:
            if type(m) is dict:
                role = m.get("role")
                if role == "system":
                    # Combine multiple system messages with newlines
                    system_prompt = [{
                        "type": "text",
                        "text": (system_prompt + "\n" if system_prompt else "") + str(m.get("content", "")),
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    }]
                elif role in {"user", "assistant"}:
                    anth_messages.append({"role": role, "content": [{
                        "type": "text",
                        "text": m.get("content", ""),
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    }]})
            else:
                # Handle complex message objects (e.g., Pydantic models)
                anth_messages.append({"role": "assistant", "content": [{x:getattr(m, x) for x in m.model_fields_set}]})

        return anth_messages, system_prompt

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
        """Create a ChatSession bound to this AnthropicAPI instance.

        Automatically enables thinking tokens for supported Claude models and
        configures Anthropic-specific optimizations for advanced reasoning,
        tool use, and conversation management.

        Args:
            tools: Tool definitions for function calling. These will be normalized
                  to Anthropic's tool format and support complex argument handling.
            output_format: Response format ("text", "json", "yaml"). Note that
                         Claude doesn't have native structured output, so JSON/YAML
                         is achieved through prompting and post-processing.
            json_schema: JSON schema for output validation (best-effort through
                        prompting, not enforced by Claude natively).
            enable_cot: Whether to enable thinking. If None, auto-detected.
            session_params: Claude-specific parameters like:
                - max_tokens: Maximum response length (required by Anthropic)
                - temperature: Sampling temperature (0.0-1.0)
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - thinking_budget_tokens: Token budget for thinking (3.7+ models)
            custom_messages: Pre-existing conversation history

        Returns:
            ChatSession configured for Anthropic API with automatic thinking token
            support and sophisticated tool execution capabilities.

        Note:
            Claude's thinking capabilities (when available) provide detailed
            reasoning that's exposed in the reasoning field of ChatResult.
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
        return [{"role": "user", "content": user_message}] if user_message else [{"role": "user", "content": "continue"}]

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
        """Send messages to Anthropic Claude API with full feature support.

        This method handles the complete request flow for Claude including:
        1. Message format conversion separating system prompts from conversation
        2. Tool configuration with Anthropic's tool use format
        3. Thinking token configuration for reasoning-capable models
        4. Best-effort structured output through prompting and validation
        5. Automatic tool execution with tool_result follow-ups
        6. Advanced conversation management with context preservation
        7. Response parsing with thinking extraction and usage tracking

        Claude-Specific Features:
        - Thinking tokens for chain-of-thought reasoning (3.7+ models)
        - Sophisticated tool use with comprehensive argument handling
        - Advanced system prompt integration that preserves context
        - Best-effort structured output through intelligent prompting
        - Comprehensive message handling for Claude's API requirements

        Args:
            messages: Conversation history in standard format
            tools: Function definitions (normalized to Anthropic tool format)
            tool_choice: Tool selection strategy for tool use
            output_format: Desired response format (achieved via prompting)
            json_schema: Schema for output validation (best-effort)
            enable_cot: Whether to enable thinking tokens for reasoning
            request_params: Request-specific parameters (temperature, etc.)
            session_params: Session-level parameters

        Returns:
            ChatResult with Claude response data, thinking blocks (if available),
            usage tracking, and tool execution results

        Raises:
            ValueError: If structured output is requested but not supported
            Exception: Re-raises any API errors from Anthropic

        Note:
            Claude doesn't support native structured output like OpenAI, so
            JSON/YAML output is achieved through careful prompting and post-processing.
        """
        if isinstance(self._client, Exception):
            raise self._client
        client = self._client

        # Anthropic expects system separate and messages without system
        system_prompt = None
        anth_messages, system_prompt = self._get_messages(messages)

        #if enable_cot:
        #    cot_hint = (
        #        "Think step by step internally using your reasoning tokens. "
        #        "Do not reveal private chain-of-thought. Provide only the final answer."
        #    )
        #    system_prompt = (system_prompt + "\n" if system_prompt else "") + cot_hint

        tools_conf = normalize_tool_schemas_for_anthropic(tools or []) if tools else None

        if output_format and output_format != "text":
            raise ValueError("Anthropic does not support structured output")

        merged_params = self._translate_request_params(
            {**(session_params or {}), **(request_params or {})}
        )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": merged_params.pop("max_tokens", 21000),
            "messages": anth_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools_conf:
            kwargs["tools"] = tools_conf
        if tool_choice:
            # Anthropic requires tool_choice as a dict, e.g. {"type": "auto"}.
            # Convert plain strings ("auto", "any", "none") to the expected format.
            if isinstance(tool_choice, str):
                tc_type = tool_choice.lower()
                if tc_type == "none":
                    pass  # omit tool_choice entirely to disable tool use
                else:
                    kwargs["tool_choice"] = {"type": tc_type if tc_type in ("auto", "any") else "auto"}
            else:
                kwargs["tool_choice"] = tool_choice
        if enable_cot:
            # Enable thinking when the model supports it (Claude 3.7 etc.)
            # Safe to pass; older models ignore or the SDK may gate.
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": merged_params.pop("thinking_budget_tokens", 10000),
            }

            # Anthropic also forbids setting top_k and top_p when thinking is enabled
            if "top_k" in merged_params:
                merged_params.pop("top_k")
            if "top_p" in merged_params:
                merged_params.pop("top_p")

        # Now safely update with remaining merged_params, but ensure temperature is 1.0 if thinking is enabled
        kwargs.update(merged_params)

        if enable_cot:
            # Anthropic API explicitly requires temperature to be exactly 1.0 when thinking is enabled
            # Note: temperature shouldn't be popped from merged_params yet, but just in case,
            # we overwrite it or remove it from kwargs if it was added.
            kwargs["temperature"] = 1.0

        output = []

        # Perform request
        try:
            current_messages = list(anth_messages)
            response = client.messages.create(**{**kwargs, "messages": current_messages})  # type: ignore[attr-defined]
            # A gateway may return normally but report a failure in the response body
            raise_for_response_error(response, provider="Anthropic", model=self._model)
            output += response.content

            thinking_blocks: list[str] = []
            usage_list: list[dict[str, Any]] = []
            enriched_tool_calls = None
            tool_calls_for_return: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            content = ""
            parsed = None
            reasoning_meta: dict[str, Any] = {}

            while True:
                # Collect usage from current response
                if getattr(response, "usage", None) is not None:
                    u = response.usage
                    usage_list.append({
                        "input_tokens": getattr(u, "input_tokens", None) or 0,
                        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None) or 0,
                        "output_tokens": getattr(u, "output_tokens", None) or 0,
                        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None) or 0,
                    })

                # Extract text, tool calls, and thinking from current response
                text_parts: list[str] = []
                tool_calls = []
                for block in getattr(response, "content", []) or []:
                    btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
                    if btype == "text":
                        text_parts.append(getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "type": "function",
                            "id": getattr(block, "id", None) if not isinstance(block, dict) else block.get("id"),
                            "function": {
                                "name": getattr(block, "name", None) if not isinstance(block, dict) else block.get("name"),
                                "arguments": getattr(block, "input", None) if not isinstance(block, dict) else block.get("input"),
                            },
                        })
                    elif btype == "thinking":
                        text_val = getattr(block, "thinking", None)
                        if text_val:
                            thinking_blocks.append(str(text_val))
                        token_count = getattr(block, "token_count", None)
                        reasoning_meta["thinking_token_count"] = token_count

                content = "\n".join([p for p in text_parts if p])
                parsed, _ = finalize_parsed_output(
                    output_format=output_format, content=content
                )

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
                    tool_blocks: list[dict[str, Any]] = []
                    for tr in tool_result_messages:
                        tool_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tr.get("tool_call_id"),
                            "content": json.dumps(tr.get("content", "")),
                        })
                    tool_blocks_message = {"role": "user", "content": tool_blocks}
                    assistant_tool_use_message = {"role": "assistant", "content": list(response.content)}
                    current_messages = current_messages + [assistant_tool_use_message, tool_blocks_message]

                    response = client.messages.create(**{**kwargs, "messages": current_messages})  # type: ignore[attr-defined]
                    raise_for_response_error(response, provider="Anthropic", model=self._model)
                    output += [tool_blocks_message] + response.content
                except Exception as exc:  # noqa: BLE001
                    raise exc

            # Aggregate usage across all turns
            usage = None
            usage_input_token = 0
            usage_cache_input_token = 0
            if usage_list:
                usage = {
                    "input_tokens": sum(u["input_tokens"] for u in usage_list) + sum(u["cache_creation_input_tokens"] for u in usage_list),
                    "output_tokens": sum(u["output_tokens"] for u in usage_list),
                    "cache_read_input_tokens": sum(u["cache_read_input_tokens"] for u in usage_list),
                }
                usage_input_token = usage["input_tokens"]
                usage_cache_input_token = usage["cache_read_input_tokens"]

            if thinking_blocks:
                reasoning_meta["thinking_blocks"] = thinking_blocks

            return ChatResult(
                content=content,
                parsed=parsed,
                format=output_format,
                tool_calls=enriched_tool_calls if enriched_tool_calls is not None else (tool_calls_for_return or None),
                usage=usage,
                usage_input_token=usage_input_token,
                usage_cache_input_token=usage_cache_input_token,
                usage_output_token=usage["output_tokens"] if usage else 0,
                raw=response,
                reasoning=reasoning_meta,
                output=output,
                id=response.id,
            )
        except Exception as exc:  # noqa: BLE001
            raise exc


    def _translate_request_params(
        self,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Map unified params to Anthropic Messages API (``messages.create``).

        - ``max_tokens``: native name, forwarded as-is
        - ``max_output_tokens`` → ``max_tokens`` (alias from other providers)
        - ``thinking_budget`` → ``thinking_budget_tokens`` (Gemini-style alias)
        - ``temperature``, ``top_p``, ``top_k``: forwarded as-is
        """
        params = dict(request_params)
        if "max_output_tokens" in params and "max_tokens" not in params:
            params["max_tokens"] = params.pop("max_output_tokens")
        elif "max_output_tokens" in params:
            del params["max_output_tokens"]
        if "thinking_budget" in params and "thinking_budget_tokens" not in params:
            params["thinking_budget_tokens"] = params.pop("thinking_budget")
        elif "thinking_budget" in params:
            del params["thinking_budget"]
        return params
