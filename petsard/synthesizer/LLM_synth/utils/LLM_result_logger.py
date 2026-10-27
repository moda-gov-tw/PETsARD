"""
LLM Result Logger - Thread-safe logging utility for LLM interactions.

This module provides a comprehensive logging system designed specifically for
Large Language Model (LLM) interactions and results. It offers thread-safe
file operations, structured logging formats, and flexible output options.

Key Features:
- Thread-safe logging operations with lock-based synchronization
- Structured format for LLM prompts, responses, and system prompts
- Automatic timestamped filename generation
- Optional console output with hierarchical verbose levels
- UTF-8 encoding support for international characters
- Organized file storage in dedicated LOG_FILES directory

Usage:
    # Basic usage
    logger = Logger("gpt-4o")
    logger.log_system_prompt("You are a helpful assistant...")
    logger.log("What is 2+2?", "2+2 equals 4")

    # With custom filename and console output (verbose 1 = standard messages)
    logger = Logger("claude", "custom_session.txt", verbose=1)
    logger.simple_log("Session started")

Design Philosophy:
The logger is designed to be simple, reliable, and focused on LLM use cases.
It prioritizes data integrity through thread-safe operations and provides
clear separation between different types of logged content.
"""

import os
import sys
import time
from threading import Lock

# Default log file storage location (overridable per Logger instance via `folder=`)
FOLDER = "LOG_FILES"

class Logger:
    """Thread-safe logger for LLM interactions with structured output formats.

    This class provides a robust logging system specifically designed for
    capturing and organizing LLM interactions. It ensures thread safety
    through lock-based synchronization and offers flexible output options.

    Attributes:
        _file_name: Full path to the log file
        _verbose: Logger console verbosity (0 = no print; N > 0 prints messages with verbose <= N)
        _lock: Threading lock for ensuring thread-safe operations
    """

    def __init__(self, model_name: str, file_name: str = None, verbose: int = 0, folder: str = None,
                 log_prompts: bool = True):
        """Initialize the Logger with configuration options.

        Args:
            model_name: Name of the LLM model being logged (used for auto-naming)
                       Path separators ('/' and '\\') will be replaced with '-'
            file_name: Optional custom filename. If None, generates timestamped name
            verbose: Console print level (default 0 = no print). A message prints when
                     verbose > 0 and logger verbose >= message verbose.
            log_prompts: If False, system prompts, prompts and responses are logged
                     (file and console) as their length only, never their text.
                     Can also be set later through the `log_prompts` attribute.

        Note:
            The LOG_FILES directory is created automatically if it doesn't exist.
            Auto-generated filenames follow the format: {sanitized_model_name}_{YYYYMMDD_HHMMSS}.txt
            Model names containing '/' or '\\' are automatically sanitized to prevent path issues.
        """
        log_folder = folder if folder is not None else FOLDER
        os.makedirs(log_folder, exist_ok=True)

        if file_name:
            self._file_name = os.path.join(log_folder, file_name)
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            sanitized_model_name = model_name.replace('/', '-').replace('\\', '-')
            self._file_name = os.path.join(log_folder, f"{sanitized_model_name}_{timestamp}.txt")

        # Store configuration
        self._verbose = verbose
        self.log_prompts = log_prompts

        # Initialize thread synchronization lock
        self._lock = Lock()

    def _should_print(self, msg_verbose: int) -> bool:
        """Return True if message should be printed to console."""
        return self._verbose > 0 and self._verbose >= msg_verbose

    def _payload(self, text):
        """Return text as logged: unchanged, or only its length when log_prompts is False."""
        if self.log_prompts:
            return text
        return f"<omitted (log_prompts: false), {0 if text is None else len(str(text))} chars>"

    @staticmethod
    def _safe_print(text: str) -> None:
        """Print text to stdout, replacing characters the console can't encode."""
        try:
            print(text)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
            print(text.encode(enc, errors="replace").decode(enc, errors="replace"))

    def log_system_prompt(self, system_prompt: str, verbose: int = 2):
        """Log a system prompt with structured formatting.

        System prompts define the behavior and context for LLM interactions.
        This method logs them in a clearly identifiable format that can be
        easily parsed and distinguished from regular conversation logs.

        Args:
            system_prompt: The system prompt text to log
            verbose: Minimum logger verbose level required to print (default 1)

        Format:
            SYSTEM_PROMPT=
            {system_prompt_content}

        Note:
            This method is thread-safe and will append to the existing log file.
            File writes are unconditional; verbose only affects console output.
            With log_prompts False only the prompt's length is written.
        """
        system_prompt = self._payload(system_prompt)
        # Acquire thread lock to ensure atomic write operation
        with self._lock:
            # Write to log file with UTF-8 encoding for international character support
            with open(self._file_name, "a", encoding="utf-8") as f:
                f.write(f"\nSYSTEM_PROMPT=\n{system_prompt}\n\n")

            if self._should_print(verbose):
                self._safe_print(f"\nSYSTEM_PROMPT=\n{system_prompt}\n\n")

    def log(self, prompt: str, response: str, prefix: str = "", verbose: int = 2):
        """Log a prompt-response pair with structured formatting.

        This is the primary logging method for capturing LLM interactions.
        It creates a clear, parseable format that separates prompts from
        responses and supports optional prefixes for categorization.

        Args:
            prompt: The input prompt sent to the LLM
            response: The output response received from the LLM
            prefix: Optional prefix for categorizing entries (e.g., "TEST_", "EVAL_")
            verbose: Minimum logger verbose level required to print (default 1)

        Format:
            {prefix}PROMPT=
            {prompt_content}

            {prefix}RESPONSE=
            {response_content}


        Note:
            The double newlines provide clear visual separation between
            different prompt-response pairs in the log file.
            With log_prompts False only the lengths of prompt and response are written.
        """
        prompt, response = self._payload(prompt), self._payload(response)
        # Acquire thread lock to ensure atomic write operation
        with self._lock:
            # Write prompt-response pair to log file
            with open(self._file_name, "a", encoding="utf-8") as f:
                f.write(f"{prefix}PROMPT=\n{prompt}\n\n{prefix}RESPONSE=\n{response}\n\n\n")

            if self._should_print(verbose):
                self._safe_print(f"{prefix}PROMPT=\n{prompt}\n\n{prefix}RESPONSE=\n{response}\n\n\n")

    def simple_log(self, message: str, verbose: int = 1):
        """Log a simple message without structured formatting.

        This method is useful for logging metadata, session information,
        timestamps, or any other unstructured content that doesn't fit
        the prompt-response pattern.

        Args:
            message: The message content to log
            verbose: Minimum logger verbose level required to print (default 1)

        Format:
            {message_content}

        Usage Examples:
            logger.simple_log("Session started at 2024-01-15 10:30:00")
            logger.simple_log("Model temperature set to 0.7")
            logger.simple_log("--- End of conversation ---")
        """
        # Acquire thread lock to ensure atomic write operation
        with self._lock:
            # Write simple message to log file
            with open(self._file_name, "a", encoding="utf-8") as f:
                f.write(f"{message}\n\n")

            if self._should_print(verbose):
                self._safe_print(f"{message}\n\n")

    def get_logs(self):
        """Retrieve the complete contents of the log file.

        This method provides read access to the entire log file content,
        which can be useful for analysis, processing, or displaying
        conversation history.

        Returns:
            str: The complete contents of the log file

        Note:
            This method is thread-safe and will wait for any ongoing
            write operations to complete before reading.
        """
        # Acquire thread lock to ensure we don't read during write operations
        with self._lock:
            # Read and return entire log file content
            with open(self._file_name, encoding="utf-8") as f:
                return f.read()

    def get_file_name(self):
        """Get the full path to the log file.

        Returns:
            str: The complete file path where logs are being written

        Usage:
            This is useful for external tools that need to process the log file,
            or for displaying the log location to users.
        """
        return self._file_name
