import requests
import json
import os
import time
import re
import ipaddress
from urllib.parse import urlparse
from typing import Callable, Optional, Dict, Any, List
from .llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
RATE_LIMIT_WAIT_TIME = 25  # seconds
DEFAULT_WAIT_TIME = 25  # seconds
DEFAULT_TIMEOUT_SECONDS = 600
PEG_NATIVE_MAX_RETRIES = 2


class BackendUnavailableError(RuntimeError):
    """Raised when an OpenAI-compatible backend cannot accept connections."""


class GenericOpenAIAPIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        api_url: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_seconds: int | None = None,
        extra_body: Optional[Dict[str, Any]] = None,
        transient_failure_recovery: Optional[Callable[[Exception], bool]] = None,
    ):
        self.api_key = api_key
        self.base_url = api_url.rstrip('/')  # Remove trailing slash if present
        self.generate_url = f"{self.base_url}/chat/completions"
        self.max_retries = max_retries
        self.timeout_seconds = int(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get("VIDEO_ANALYZER_TEXT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        )
        self.extra_body = dict(extra_body or {})
        self.transient_failure_recovery = transient_failure_recovery
        self.session = requests.Session()
        if self._should_bypass_env_proxy():
            self.session.trust_env = False

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256,
        image_paths: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None) -> Dict[Any, Any]:
        """Generate response from OpenAI-compatible API."""
        # Prepare request content
        paths = image_paths or ([image_path] if image_path else [])
        if paths:
            content = [{"type": "text", "text": prompt}]
            for path in paths:
                base64_image = self.encode_image(path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                )
        else:
            content = prompt

        # Prepare request data
        data = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
            "temperature": temperature,
            "max_tokens": num_predict
        }
        if self.extra_body:
            data.update(self.extra_body)
        if extra_body:
            data.update(extra_body)

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/byjlw/video-analyzer",
            "X-Title": "Video Analyzer",
            "Content-Type": "application/json"
        }

        recovery_attempted = False
        peg_native_retries = 0

        # Try request with retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.generate_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 400:
                    detail = response.text[:1000]
                    raise requests.exceptions.HTTPError(
                        f"{response.status_code} {detail}",
                        response=response,
                    )
                
                # Parse successful response
                try:
                    json_response = response.json()
                    if 'error' in json_response:
                        raise Exception(f"API error: {json_response['error']}")
                    
                    if stream:
                        return self._handle_streaming_response(response)
                    
                    if 'choices' not in json_response or not json_response['choices']:
                        raise Exception("No choices in response")
                        
                    message = json_response['choices'][0].get('message', {})
                    if not message:
                        raise Exception("No content in response message")

                    content = message.get("content")
                    source = "content"
                    if not content and message.get("reasoning_content") and self._allows_reasoning_content_fallback():
                        content = self._clean_reasoning_content(message.get("reasoning_content", ""))
                        source = "reasoning_content"
                        logger.warning("API response content was empty; using reasoning_content fallback")
                    elif not content and message.get("reasoning_content"):
                        raise Exception("Response content is empty; reasoning_content fallback is only allowed for local/LAN endpoints")
                    if content is None:
                        raise Exception("No content in response message")

                    return {"response": content, "response_source": source}
                    
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response: {response.text}")
                    
            except Exception as e:
                if self._is_peg_native_format_error(e):
                    if (
                        peg_native_retries >= PEG_NATIVE_MAX_RETRIES
                        or attempt == self.max_retries - 1
                    ):
                        raise Exception(f"An error occurred: {str(e)}") from e
                    peg_native_retries += 1
                    logger.warning(
                        "Model output did not match peg-native format; retrying immediately (%s/%s)",
                        peg_native_retries,
                        PEG_NATIVE_MAX_RETRIES,
                    )
                    continue
                if attempt == self.max_retries - 1:  # Last attempt
                    raise Exception(f"An error occurred: {str(e)}")

                if (
                    not recovery_attempted
                    and self.transient_failure_recovery is not None
                    and self._is_recoverable_connection_error(e)
                ):
                    recovery_attempted = True
                    try:
                        if self.transient_failure_recovery(e):
                            logger.warning("Recovered local backend after connection interruption; retrying now")
                            continue
                    except Exception as recovery_error:
                        logger.warning("Local backend recovery failed: %s", recovery_error)

                if self._is_recoverable_connection_error(e):
                    raise BackendUnavailableError(
                        f"Backend unavailable at {self.base_url}: {e}"
                    ) from e

                # Get wait time based on error
                wait_time = RATE_LIMIT_WAIT_TIME
                if isinstance(e, requests.exceptions.HTTPError) and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise Exception(f"An error occurred: {str(e)}")

                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                    # Try to get wait time from Retry-After header
                    if 'Retry-After' in e.response.headers:
                        try:
                            wait_time = int(e.response.headers['Retry-After'])
                            logger.info(f"Using Retry-After header value: {wait_time} seconds")
                        except (ValueError, TypeError):
                            logger.warning("Invalid Retry-After header value, using default wait time")
                else:
                    wait_time = DEFAULT_WAIT_TIME
                
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                logger.warning(f"Waiting {wait_time} seconds before retry")
                time.sleep(wait_time)

    @staticmethod
    def _is_recoverable_connection_error(error: Exception) -> bool:
        return isinstance(
            error,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ),
        )

    @staticmethod
    def _is_peg_native_format_error(error: Exception) -> bool:
        return (
            isinstance(error, requests.exceptions.HTTPError)
            and error.response is not None
            and error.response.status_code >= 500
            and "does not match the expected peg-native format" in str(error)
        )

    def _allows_reasoning_content_fallback(self) -> bool:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or ""
        if host in {"localhost", "127.0.0.1", "::1"}:
            return True
        try:
            address = ipaddress.ip_address(host)
            return address.is_private or address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")
        except ValueError:
            return host.endswith(".local") or host.endswith(".lan") or host.endswith(".taild500c8.ts.net")

    def _should_bypass_env_proxy(self) -> bool:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or ""
        if host in {"localhost", "127.0.0.1", "::1"}:
            return True
        try:
            address = ipaddress.ip_address(host)
            return address.is_private or address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")
        except ValueError:
            return host.endswith(".local") or host.endswith(".lan") or host.endswith(".taild500c8.ts.net")

    @staticmethod
    def _clean_reasoning_content(text: str) -> str:
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        """Handle streaming response from API.
        
        Args:
            response: Streaming response from API
            
        Returns:
            Dict containing accumulated response
        """
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'choices' in json_response and len(json_response['choices']) > 0:
                        delta = json_response['choices'][0].get('delta', {})
                        if 'content' in delta:
                            accumulated_response += delta['content']
                except json.JSONDecodeError:
                    continue

        return {"response": accumulated_response}
