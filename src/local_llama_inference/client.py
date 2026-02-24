"""HTTP REST client for llama-server."""

import json
import httpx
from typing import Optional, Dict, Any, List, Iterator

from .exceptions import APIError, ClientError


class LlamaClient:
    """Synchronous HTTP client for llama-server REST API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_key: Optional[str] = None,
        timeout: float = 600.0,
    ):
        """
        Initialize client.

        Args:
            base_url: Base URL of llama-server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(headers=self._headers, timeout=timeout)

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> httpx.Response:
        """Make HTTP request."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            try:
                data = e.response.json()
                msg = data.get("error", {}).get("message", str(e))
            except:
                msg = str(e)
            raise APIError(e.response.status_code, msg) from e
        except httpx.RequestError as e:
            raise ClientError(str(e)) from e

    # Health & Info Endpoints
    def health(self) -> Dict[str, Any]:
        """GET /health - Server health status."""
        return self._request("GET", "/health").json()

    def get_props(self) -> Dict[str, Any]:
        """GET /props - Server properties."""
        return self._request("GET", "/props").json()

    def set_props(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """POST /props - Update server properties."""
        return self._request("POST", "/props", json=props).json()

    def get_models(self) -> Dict[str, Any]:
        """GET /models - List loaded models."""
        return self._request("GET", "/models").json()

    def get_slots(self, fail_on_no_slot: bool = False) -> List[Dict]:
        """GET /slots - Get inference slots status."""
        params = {"fail_on_no_slot": fail_on_no_slot}
        return self._request("GET", "/slots", params=params).json()

    def get_metrics(self) -> str:
        """GET /metrics - Prometheus metrics."""
        return self._request("GET", "/metrics").text

    def get_lora_adapters(self) -> List[Dict]:
        """GET /lora-adapters - List loaded LoRA adapters."""
        return self._request("GET", "/lora-adapters").json()

    def set_lora_adapters(self, adapters: List[Dict]) -> Dict:
        """POST /lora-adapters - Set LoRA adapter scales."""
        return self._request("POST", "/lora-adapters", json=adapters).json()

    # Text Generation Endpoints
    def complete(self, prompt: str, stream: bool = False, **kwargs) -> Dict | Iterator:
        """
        POST /completion - Text completion.

        Args:
            prompt: Text prompt
            stream: Stream response tokens
            **kwargs: Additional parameters (temperature, top_p, etc.)

        Returns:
            Dict if stream=False, Iterator[str] if stream=True
        """
        payload = {"prompt": prompt, "stream": stream, **kwargs}
        if stream:
            return self._stream_response(self._request("POST", "/completion", json=payload))
        return self._request("POST", "/completion", json=payload).json()

    def chat(self, messages: List[Dict], stream: bool = False, **kwargs) -> Dict | Iterator:
        """
        POST /v1/chat/completions - Chat completion (OpenAI-compatible).

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Stream response
            **kwargs: Additional parameters

        Returns:
            Dict if stream=False, Iterator[str] if stream=True
        """
        payload = {"messages": messages, "stream": stream, **kwargs}
        if stream:
            return self._stream_response(
                self._request("POST", "/v1/chat/completions", json=payload)
            )
        return self._request("POST", "/v1/chat/completions", json=payload).json()

    def stream_chat(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """Stream chat completion tokens."""
        yield from self.chat(messages, stream=True, **kwargs)

    def stream_complete(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream completion tokens."""
        yield from self.complete(prompt, stream=True, **kwargs)

    def infill(self, prefix: str, suffix: str, **kwargs) -> Dict:
        """POST /infill - Fill-in-the-middle code completion."""
        payload = {"input_prefix": prefix, "input_suffix": suffix, **kwargs}
        return self._request("POST", "/infill", json=payload).json()

    # Embeddings & Reranking
    def embed(self, input: str | List[str], **kwargs) -> Dict:
        """POST /v1/embeddings - Generate embeddings."""
        payload = {"input": input, **kwargs}
        return self._request("POST", "/v1/embeddings", json=payload).json()

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> Dict:
        """POST /v1/rerank - Rerank documents."""
        payload = {"query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        return self._request("POST", "/v1/rerank", json=payload).json()

    # Token Utilities
    def tokenize(
        self,
        content: str,
        add_special: bool = True,
        parse_special: bool = False,
        with_pieces: bool = False,
    ) -> Dict:
        """POST /tokenize - Tokenize text."""
        payload = {
            "content": content,
            "add_special": add_special,
            "parse_special": parse_special,
            "with_pieces": with_pieces,
        }
        return self._request("POST", "/tokenize", json=payload).json()

    def detokenize(self, tokens: List[int]) -> Dict:
        """POST /detokenize - Detokenize token IDs."""
        return self._request("POST", "/detokenize", json={"tokens": tokens}).json()

    def apply_template(self, messages: List[Dict]) -> Dict:
        """POST /apply-template - Apply chat template."""
        return self._request("POST", "/apply-template", json={"messages": messages}).json()

    # Model Management (Router Mode)
    def load_model(self, model_name: str) -> Dict:
        """POST /models/load - Load a model."""
        return self._request("POST", "/models/load", json={"model": model_name}).json()

    def unload_model(self, model_name: str) -> Dict:
        """POST /models/unload - Unload a model."""
        return self._request("POST", "/models/unload", json={"model": model_name}).json()

    # Slot Management
    def save_slot(self, slot_id: int, filename: str) -> Dict:
        """POST /slots/{slot_id}?action=save - Save KV cache."""
        return self._request(
            "POST",
            f"/slots/{slot_id}",
            params={"action": "save"},
            json={"filename": filename},
        ).json()

    def restore_slot(self, slot_id: int, filename: str) -> Dict:
        """POST /slots/{slot_id}?action=restore - Restore KV cache."""
        return self._request(
            "POST",
            f"/slots/{slot_id}",
            params={"action": "restore"},
            json={"filename": filename},
        ).json()

    def erase_slot(self, slot_id: int) -> Dict:
        """POST /slots/{slot_id}?action=erase - Erase KV cache."""
        return self._request("POST", f"/slots/{slot_id}", params={"action": "erase"}).json()

    def _stream_response(self, response: httpx.Response) -> Iterator[str]:
        """Parse SSE stream response."""
        with response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if content := delta.get("content", ""):
                            yield content
                    except json.JSONDecodeError:
                        continue

    def close(self):
        """Close HTTP connection."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"LlamaClient({self.base_url})"
