"""Google Gemini API implementation built on APITemplate.

This module provides a comprehensive integration with Google's Gemini models
through the latest Google AI SDK, offering advanced capabilities while
maintaining compatibility with the unified API interface.

Key Features:
- Native function calling via function_declarations with automatic execution
- Advanced structured output with native MIME type and schema support
- YAML output support with intelligent MIME type handling
- Automatic reasoning capabilities for Gemini thinking models
- Multi-modal support foundation (text, with extensibility for images/audio)
- Comprehensive tool execution with automatic follow-up conversations
- Advanced generation configuration with safety settings

Google Gemini-Specific Capabilities:
- Native JSON and YAML MIME type support for structured outputs
- Function declarations format for sophisticated tool calling
- Advanced generation configuration with temperature, top-p, top-k controls
- Safety settings and content filtering integration
- Thinking configuration for reasoning-capable models (Gemini Flash Thinking)
- Multi-turn conversation optimization with context preservation
- Token usage tracking with detailed prompt/candidate/total breakdowns

Supported Models:
- Gemini 1.5 Pro (recommended for complex tasks)
- Gemini 1.5 Flash (fastest, cost-effective)
- Gemini 2.5 models (when available)
- Gemini Flash Thinking (reasoning capabilities)
- Custom fine-tuned Gemini models

Authentication:
Requires a valid Google AI API key. Ensure your key has access to the Gemini
models you intend to use and is properly configured for your project.

Multi-Modal Foundation:
While this implementation focuses on text interactions, it's built on Google's
multi-modal architecture and can be extended to support image, audio, and
video inputs in future versions.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from .base import (
    APIResponseError,
    APITemplate,
    ChatResult,
    ChatSession,
    _safe_json_dumps,
    execute_registered_functions,
    finalize_parsed_output,
    normalize_tool_schemas_for_gemini,
    raise_for_response_error,
)


class GeminiAPI(APITemplate):
    """Google Gemini API implementation with full multi-modal and reasoning support.

    This class provides a complete integration with Google's Gemini models using
    the latest Google AI SDK. It supports advanced features like native structured
    output, sophisticated function calling, reasoning capabilities, and provides
    a foundation for multi-modal interactions.

    Key Implementation Features:
    - Native MIME type support for JSON/YAML structured outputs
    - Advanced function_declarations format for tool calling
    - Automatic reasoning detection and configuration for thinking models
    - Comprehensive conversation management with Google's Content format
    - Advanced generation configuration with safety and performance tuning
    - Multi-modal content type support (extensible to images/audio/video)
    """

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Initialize Google Gemini API client.

        Args:
            api_key: Your Google AI API key
            model: Gemini model to use. Defaults to "gemini-1.5-pro" for
                  the best balance of capability and performance.
                  Popular options:
                  - "gemini-1.5-pro" (recommended default, most capable)
                  - "gemini-1.5-flash" (fastest, most cost-effective)
                  - "gemini-2.5-flash-exp" (experimental, latest features)
                  - "gemini-flash-thinking" (reasoning capabilities)
        """
        super().__init__(api_key=api_key, model=model or "gemini-1.5-pro")
        self._client = None
        self._ensure_client()

    def _ensure_client(self) -> None:
        """Initialize the Google GenAI client with lazy loading and error handling.

        This method uses lazy initialization to avoid import errors when the
        Google AI library is not installed. Any initialization errors are
        stored and raised later during actual API calls, providing better
        error messaging and graceful handling of missing dependencies.
        """
        if self._client is not None:
            return
        try:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001
            self._client = exc

    def _supports_cot(self) -> bool:
        """Detect if the current Gemini model supports chain-of-thought reasoning.

        Returns:
            True if the model supports reasoning capabilities with thinking configuration.

        Note:
            Gemini reasoning models typically include "thinking" or "reasoning" in
            their names, or are part of the newer 2.5/3 model families which
            have built-in reasoning capabilities. This detection enables automatic
            configuration of thinking settings and reasoning prompts.
        """
        name = (self._model or "").lower()
        # Gemini reasoning models commonly include "flash-thinking" or similar
        return (
            any(tag in name for tag in ["thinking", "reasoning"])
            or "2.5" in name
            or "-3-" in name
        )

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
        """Create a ChatSession bound to this GeminiAPI instance.

        Automatically enables reasoning capabilities for supported Gemini models
        and configures Google-specific optimizations for multi-modal and
        structured output capabilities.

        Args:
            tools: Tool definitions for function calling. These will be converted
                  to Gemini's function_declarations format automatically.
            output_format: Response format ("text", "json", "yaml"). Gemini has
                         native support for JSON via response_mime_type.
            json_schema: JSON schema for structured output validation. Used with
                        Gemini's native response_schema when output_format="json".
            enable_cot: Whether to enable thinking. If None, auto-detected.
            session_params: Unified parameters (translated before send), e.g.:
                - max_tokens: Output cap (→ max_output_tokens)
                - temperature, top_p, top_k: Sampling controls
                - thinking_level: Reasoning depth ("LOW", "MEDIUM", "HIGH")
                - thinking_budget or thinking_budget_tokens: Reasoning token budget
                - safety_settings: Content filtering configuration
                - generation_config: Advanced generation settings
            custom_messages: Pre-existing conversation history

        Returns:
            ChatSession configured for Gemini API with automatic reasoning detection
            and native structured output capabilities.

        Note:
            Gemini's multi-modal capabilities can be leveraged by extending the
            Content format in future versions to include image/audio/video parts.
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
        return [{"role": "user", "content": user_message}] if user_message else [{"role": "user", "content": ""}]

    def _raise_for_response_error(self, resp: Any) -> None:
        """Raise APIResponseError if Gemini returned no usable candidate content.

        Gemini reports blocked prompts and aborted generations inside the response
        (``prompt_feedback.block_reason`` / ``candidates[0].finish_reason``) instead
        of raising, leaving ``resp.parts`` as None.
        """
        raise_for_response_error(resp, provider="Gemini", model=self._model)

        if getattr(resp, "parts", None) is not None:
            return

        feedback = getattr(resp, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            reason, status = block_reason, "blocked"
            message = getattr(feedback, "block_reason_message", None) or "prompt was blocked"
        else:
            candidates = getattr(resp, "candidates", None) or []
            first = candidates[0] if candidates else None
            reason, status = getattr(first, "finish_reason", None), "empty"
            message = getattr(first, "finish_message", None) or "response contains no candidate content"

        raise APIResponseError(
            str(message),
            provider="Gemini",
            code=(getattr(reason, "name", None) or str(reason)) if reason else None,
            status=status,
            model=self._model,
            response=resp,
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
        """Send messages to Google Gemini API with full feature support.

        This method handles the complete request flow for Gemini including:
        1. Message format conversion to Google's Content structure
        2. Tool configuration using function_declarations format
        3. Native structured output via response_mime_type and response_schema
        4. Reasoning/thinking configuration for capable models
        5. Automatic tool execution with function_response follow-ups
        6. Multi-modal content handling (text-focused with extensibility)
        7. Response parsing with usage tracking and reasoning extraction

        Gemini-Specific Features:
        - Native JSON/YAML MIME type support for structured outputs
        - Function declarations for sophisticated tool calling
        - Advanced generation configuration with safety settings
        - Thinking configuration for reasoning models
        - Multi-turn optimization with context preservation

        Args:
            messages: Conversation history in standard format
            tools: Function definitions (converted to function_declarations)
            tool_choice: Tool selection strategy (may be ignored by Gemini)
            output_format: Desired response format with native support
            json_schema: Schema for structured output validation
            enable_cot: Whether to enable thinking for reasoning models
            request_params: Request-specific parameters
            session_params: Session-level parameters

        Returns:
            ChatResult with Gemini response data, usage tracking, and any
            reasoning information from thinking-capable models

        Raises:
            Exception: Re-raises any API errors from Google Gemini
        """
        if isinstance(self._client, Exception):
            raise self._client
        client = self._client

        # Prepare tools
        tool_blocks, other_tools = normalize_tool_schemas_for_gemini(tools or []) if tools else (None, None)
        tool_for_config = None
        if tool_blocks:
            tool_for_config = types.Tool(function_declarations=tool_blocks)

        # Prepare generation config and system
        system_prompt = None
        contents: list[Any] = []
        output = []
        for m in messages:
            if type(m) is dict:
                role = m.get("role")
                if role == "system":
                    system_prompt = (system_prompt + "\n" if system_prompt else "") + str(
                        m.get("content", "")
                    )
                elif role in {"user", "assistant"}:
                    # New SDK expects roles "user" or "model" for history
                    mapped_role = "user" if role == "user" else "model"
                    text_part = str(m.get("content", ""))
                    contents.append(
                        types.Content(
                            role=mapped_role,
                            parts=[types.Part(text=text_part)],
                        )
                )
            else:
                contents.append(m)

        # Output format settings
        response_mime_type = None
        response_schema = None
        if output_format == "json":
            response_mime_type = "application/json"
            if json_schema:
                response_schema = json_schema
        elif output_format == "yaml":
            # Not formally supported everywhere; try MIME hint and parse later
            response_mime_type = "application/yaml"

        merged_params = self._translate_request_params(
            {**(session_params or {}), **(request_params or {})}
        )

        generation_config: dict[str, Any] = {
            "temperature": merged_params.pop("temperature", 1.0),
        }
        if response_mime_type:
            generation_config["response_mime_type"] = response_mime_type
        if response_schema:
            generation_config["response_schema"] = response_schema

        thinking_config = None
        if enable_cot or "thinking_level" in merged_params or "thinking_budget" in merged_params:
            tc_kwargs = {}
            if enable_cot:
                tc_kwargs["include_thoughts"] = True
            if "thinking_level" in merged_params:
                tc_kwargs["thinking_level"] = merged_params.get("thinking_level")
            if "thinking_budget" in merged_params:
                tc_kwargs["thinking_budget"] = merged_params.get("thinking_budget")
            thinking_config = types.ThinkingConfig(**tc_kwargs)

        # Start chat and send last user turn only (history includes earlier turns)
        try:
            # Build request config for the new SDK
            config = types.GenerateContentConfig(
                temperature=generation_config.get("temperature"),
                top_p=merged_params.get("top_p"),
                top_k=merged_params.get("top_k"),
                max_output_tokens=merged_params.get("max_output_tokens"),
                stop_sequences=merged_params.get("stop_sequences"),
                presence_penalty=merged_params.get("presence_penalty"),
                frequency_penalty=merged_params.get("frequency_penalty"),
                response_mime_type=generation_config.get("response_mime_type"),
                response_schema=generation_config.get("response_schema"),
                system_instruction=system_prompt,
                tools=(other_tools if other_tools else [])+([tool_for_config,] if tool_for_config else []),
                thinking_config=thinking_config,
                safety_settings=merged_params.get("safety_settings"),
            )

            # Send the whole conversation contents in one call
            resp = client.models.generate_content(
                model=self._model,
                contents=contents if contents else "",
                config=config,
            )
            self._raise_for_response_error(resp)
            output += resp.parts

            text = resp.text or ""
            thinking_blocks: list[str] = []
            usage_list: list[dict[str, Any]] = []
            enriched_tool_calls = None
            tool_calls_for_return: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            parsed = None
            reasoning_meta = None

            while True:
                # Collect usage from current resp
                um = getattr(resp, "usage_metadata", None)
                if um is not None:
                    cached = getattr(um, "cached_content_token_count", None) or 0
                    usage_list.append({
                        "prompt_tokens": getattr(um, "prompt_token_count", None) or 0,
                        "candidates_tokens": getattr(um, "candidates_token_count", None) or 0,
                        "thoughts_tokens": getattr(um, "thoughts_token_count", None) or 0,
                        "total_tokens": getattr(um, "total_token_count", None) or 0,
                        "input_tokens": (getattr(um, "prompt_token_count", 0) or 0) - cached,
                        "cached_content_token_count": cached,
                    })

                # Extract tool calls and thinking from current resp
                tool_calls = []
                for cand in getattr(resp, "candidates", []) or []:
                    content_obj = getattr(cand, "content", None)
                    if not content_obj:
                        continue
                    for part in getattr(content_obj, "parts", []) or []:
                        fc = getattr(part, "function_call", None)
                        if fc:
                            try:
                                if isinstance(fc, dict):
                                    fc_name = fc.get("name")
                                    fc_args = fc.get("args") or fc.get("arguments")
                                else:
                                    fc_name = getattr(fc, "name", None)
                                    fc_args = getattr(fc, "args", None)
                                    if fc_args is None:
                                        fc_args = getattr(fc, "arguments", None)
                            except Exception:
                                fc_name, fc_args = None, None
                            tool_calls.append({
                                "type": "function",
                                "function": {"name": fc_name, "arguments": fc_args},
                            })
                        # Gemini may expose reasoning parts; capture them if present
                        reasoning = getattr(part, "thought", False)
                        if reasoning:
                            try:
                                thinking_blocks.append(str(part.text))
                            except Exception:
                                pass

                parsed, reasoning_meta = finalize_parsed_output(
                    output_format=output_format, content=text
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
                    followup_contents = contents + output
                    followup_parts: list[Any] = []
                    for tr in tool_result_messages:
                        tool_name = tr.get("name")
                        tool_content = tr.get("content")
                        if not isinstance(tool_content, str):
                            tool_content = _safe_json_dumps(tool_content)
                        followup_parts.append(
                            types.Part(
                                function_response={
                                    "name": tool_name,
                                    "response": {
                                        "name": tool_name,
                                        "content": [{"text": tool_content}],
                                    },
                                }
                            )
                        )
                    followup_contents.append(
                        types.Content(role="user", parts=followup_parts)
                    )
                    resp = client.models.generate_content(
                        model=self._model,
                        contents=followup_contents,
                        config=config,
                    )
                    self._raise_for_response_error(resp)
                    output.append(followup_contents[-1])
                    output += resp.parts
                    text = resp.text or ""
                except Exception as exc:  # noqa: BLE001
                    raise exc

            # Aggregate usage across all turns
            usage = None
            usage_input_token = 0
            usage_cache_input_token = 0
            if usage_list:
                usage = {
                    "prompt_tokens": sum(u["prompt_tokens"] for u in usage_list),
                    "candidates_tokens": sum(u["candidates_tokens"] for u in usage_list) + sum(u["thoughts_tokens"] for u in usage_list),
                    "total_tokens": sum(u["total_tokens"] for u in usage_list),
                    "input_tokens": sum(u["input_tokens"] for u in usage_list),
                    "cached_content_token_count": sum(u["cached_content_token_count"] for u in usage_list),
                }
                usage_input_token = usage["input_tokens"]
                usage_cache_input_token = usage["cached_content_token_count"]

            if thinking_blocks:
                if reasoning_meta is None:
                    reasoning_meta = {}
                reasoning_meta["thinking_blocks"] = thinking_blocks

            return ChatResult(
                content=text,
                parsed=parsed,
                format=output_format,
                tool_calls=enriched_tool_calls if enriched_tool_calls is not None else (tool_calls_for_return or None),
                usage=usage,
                usage_input_token=usage_input_token,
                usage_cache_input_token=usage_cache_input_token,
                usage_output_token=usage["candidates_tokens"] if usage else 0,
                raw=resp,
                reasoning=reasoning_meta,
                output=output,
                id=resp.response_id
            )
        except Exception as exc:  # noqa: BLE001
            raise exc


    def _translate_request_params(
        self,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Map unified params to Gemini ``GenerateContentConfig`` fields.

        - ``max_tokens`` → ``max_output_tokens``
        - ``thinking_budget_tokens`` → ``thinking_budget`` (Anthropic-style alias)
        - ``temperature``, ``top_p``, ``top_k``, ``thinking_level``: as-is
        """
        params = dict(request_params)
        if "max_tokens" in params and "max_output_tokens" not in params:
            params["max_output_tokens"] = params.pop("max_tokens")
        elif "max_tokens" in params:
            del params["max_tokens"]
        if "thinking_budget_tokens" in params and "thinking_budget" not in params:
            params["thinking_budget"] = params.pop("thinking_budget_tokens")
        elif "thinking_budget_tokens" in params:
            del params["thinking_budget_tokens"]
        return params
