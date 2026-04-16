"""
Unified LLM client for Anthropic (Claude), OpenAI (GPT), Google (Gemini),
HuggingFace local models, and Ollama (local inference server).

All clients share a common interface via BaseLLMClient:
    client = get_llm_client(provider="anthropic", model="claude-sonnet-4-5")
    text = client.generate("Your prompt here")
    stats = client.get_stats()   # token counts and cost estimation

API keys are read from environment variables:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import os


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2000):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_count = 0
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from prompt."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            'model': self.model,
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
        }


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        super().__init__(model, api_key, temperature, max_tokens)

        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        # Initialize client
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key.")

        self.client = anthropic.Anthropic(api_key=api_key)

        # Token-based rate limiting (30k tokens/min limit)
        # Track tokens used in current minute window
        import time
        self.token_window_start = time.time()
        self.tokens_in_window = 0
        self.max_tokens_per_minute = 25000  # Stay under 30k limit

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from Claude with token-based rate limiting."""
        import time
        from anthropic import RateLimitError

        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        # Estimate prompt tokens (rough: 4 chars per token)
        estimated_tokens = len(prompt) // 4 + max_tokens

        # Check if we need to wait for window reset
        elapsed = time.time() - self.token_window_start
        if elapsed >= 60:
            # Reset window
            self.token_window_start = time.time()
            self.tokens_in_window = 0
        elif self.tokens_in_window + estimated_tokens > self.max_tokens_per_minute:
            # Wait for window to reset
            wait_time = 60 - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self.token_window_start = time.time()
            self.tokens_in_window = 0

        # Retry logic for rate limits, connection errors, and transient server errors
        from anthropic import APIConnectionError, InternalServerError
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                # Update stats and token tracking
                self.call_count += 1
                actual_tokens = response.usage.input_tokens + response.usage.output_tokens
                self.total_tokens += actual_tokens
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                self.tokens_in_window += actual_tokens

                if not response.content:
                    if attempt < max_retries - 1:
                        wait_time = 15 * (2 ** attempt)
                        print(f"\n  Empty response (stop_reason={response.stop_reason}), retrying in {wait_time}s ({attempt + 2}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Empty response after {max_retries} attempts (stop_reason={response.stop_reason})")
                return response.content[0].text

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 60
                    print(f"\n  Rate limit hit, waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    self.token_window_start = time.time()
                    self.tokens_in_window = 0
                else:
                    raise
            except (APIConnectionError, InternalServerError):
                if attempt < max_retries - 1:
                    wait_time = 15 * (2 ** attempt)  # 15s, 30s, 60s, 120s
                    print(f"\n  Transient error, retrying in {wait_time}s ({attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(self, model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        super().__init__(model, api_key, temperature, max_tokens)

        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Initialize client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from GPT."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Update stats
        self.call_count += 1
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.total_tokens += response.usage.total_tokens

        return response.choices[0].message.content


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(self, model: str = "gemini-2.0-flash",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        super().__init__(model, api_key, temperature, max_tokens)

        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

        # Initialize client
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key.")

        self.genai.configure(api_key=api_key)
        self.client = self.genai.GenerativeModel(model)

        # Rate limiting for paid tier (appears to be lower than documented)
        # Use 1.0s to be conservative: 60s / 1.0s = 60 req/min
        import time
        self.last_request_time = 0
        self.min_request_interval = 1.0

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from Gemini with rate limiting and retry logic."""
        import time
        from google.api_core.exceptions import ResourceExhausted

        # Rate limiting: ensure minimum interval between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        # Configure generation
        generation_config = self.genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Retry logic with exponential backoff for rate limits
        max_retries = 5
        base_delay = 30  # Start with 30s wait for rate limit recovery

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                self.last_request_time = time.time()

                # Update stats
                self.call_count += 1
                if hasattr(response, 'usage_metadata'):
                    inp = response.usage_metadata.prompt_token_count
                    out = response.usage_metadata.candidates_token_count
                    self.total_input_tokens += inp
                    self.total_output_tokens += out
                    self.total_tokens += inp + out

                return response.text

            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 15s, 30s, 60s
                    wait_time = base_delay * (2 ** attempt)
                    print(f"  Rate limit hit, waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                    self.last_request_time = time.time()
                else:
                    # Final attempt failed, re-raise
                    raise


class HuggingFaceClient(BaseLLMClient):
    """Client for HuggingFace local models."""

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2000,
                 device: str = "auto"):
        super().__init__(model, api_key, temperature, max_tokens)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.torch = torch
            self.pipeline = pipeline
        except ImportError:
            raise ImportError(
                "transformers and torch not installed. "
                "Run: pip install transformers torch accelerate"
            )

        print(f"Loading HuggingFace model: {model}")
        print("This may take a few minutes on first load...")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )

        # Create pipeline
        self.pipe = self.pipeline(
            "text-generation",
            model=self.model_obj,
            tokenizer=self.tokenizer,
            device_map=device
        )

        print(f"Model loaded successfully on device: {device}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from HuggingFace model."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        # For instruct models, format the prompt
        if "instruct" in self.model.lower() or "chat" in self.model.lower():
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Generate
        outputs = self.pipe(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Extract generated text
        generated_text = outputs[0]["generated_text"]

        # Remove the prompt from output (for non-instruct models)
        if not ("instruct" in self.model.lower() or "chat" in self.model.lower()):
            generated_text = generated_text[len(formatted_prompt):].strip()
        else:
            # For instruct models, extract only the assistant response
            if formatted_prompt in generated_text:
                generated_text = generated_text[len(formatted_prompt):].strip()

        # Update stats
        self.call_count += 1
        # Approximate token count
        self.total_tokens += len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(generated_text))

        return generated_text


class OllamaClient(BaseLLMClient):
    """Client for local Ollama models (Llama, Mistral, etc.)."""

    def __init__(self, model: str = "llama3.1:8b",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.0,
                 max_tokens: int = 512):
        super().__init__(model, api_key=None, temperature=temperature, max_tokens=max_tokens)
        self.host = host.rstrip('/')

        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests not installed. Run: pip install requests")

        # Verify server is reachable
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.host}. "
                f"Make sure Ollama is running: ollama serve\n"
                f"Original error: {e}"
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from a local Ollama model."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        response = self.requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        self.call_count += 1
        self.total_tokens += result.get('prompt_eval_count', 0) + result.get('eval_count', 0)

        return result['response']

    def list_models(self):
        """Return the list of models currently available in Ollama."""
        resp = self.requests.get(f"{self.host}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m['name'] for m in resp.json().get('models', [])]


def get_llm_client(provider: str, model: str, api_key: Optional[str] = None,
                   **kwargs) -> BaseLLMClient:
    """
    Factory function to get LLM client.

    Args:
        provider: "anthropic", "openai", "gemini", or "huggingface"
        model: Model name (e.g., "gpt-4", "claude-3-5-sonnet-20241022",
              "gemini-2.0-flash", "gemini-2.5-flash", "gemini-3-pro-preview",
              "meta-llama/Llama-3.1-8B-Instruct")
        api_key: API key (optional, can use env var)
        **kwargs: Additional arguments (temperature, max_tokens, device, etc.)

    Returns:
        LLM client instance
    """
    provider = provider.lower()

    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key, **kwargs)
    elif provider == "openai":
        return OpenAIClient(model=model, api_key=api_key, **kwargs)
    elif provider == "gemini":
        return GeminiClient(model=model, api_key=api_key, **kwargs)
    elif provider == "huggingface":
        return HuggingFaceClient(model=model, api_key=api_key, **kwargs)
    elif provider == "ollama":
        return OllamaClient(model=model, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Use 'anthropic', 'openai', 'gemini', 'huggingface', or 'ollama'."
        )
