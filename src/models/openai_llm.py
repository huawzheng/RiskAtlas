#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Compatible Interface LLM Implementation
Used to connect to vLLM OpenAI-compat service (http://host:port/v1)
"""
from typing import List, Dict, Any, Optional
import logging
import requests
import threading
import time

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


class AdaptiveConcurrencyLimiter:
    """
    Adaptive Concurrency Limiter - Dynamically adjusts concurrency based on request success/failure
    
    Principle:
    - Slowly increase concurrency on success (explore higher concurrency)
    - Quickly decrease concurrency on failure/timeout (protect server)
    - Automatically find the maximum concurrency vLLM can handle
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Concurrency control parameters
        self.min_concurrent = 2           # Minimum concurrency
        self.max_concurrent = 50          # Maximum concurrency limit
        self.current_limit = 10           # Current concurrency limit (starting value)
        
        # Adaptive parameters
        self.success_count = 0            # Consecutive success count
        self.success_threshold = 20       # Consecutive successes before increasing concurrency
        self.increase_step = 2            # Concurrency increase per step
        self.decrease_ratio = 0.5         # Decrease ratio on failure
        
        # Semaphore and lock
        self._semaphore = threading.Semaphore(self.current_limit)
        self._adjust_lock = threading.Lock()
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.last_adjust_time = time.time()
        
        self._initialized = True
        logger.info(f"🎛️  Adaptive concurrency limiter initialized: starting concurrency={self.current_limit}, range=[{self.min_concurrent}, {self.max_concurrent}]")
    
    def acquire(self, timeout: float = 300) -> bool:
        """Acquire execution permit (blocks until slot available)"""
        return self._semaphore.acquire(timeout=timeout)
    
    def release(self, success: bool = True):
        """Release execution permit and adjust concurrency based on result"""
        self._semaphore.release()
        
        with self._adjust_lock:
            self.total_requests += 1
            
            if success:
                self.success_count += 1
                # Consecutive successes reached threshold, try to increase concurrency
                if self.success_count >= self.success_threshold:
                    self._increase_limit()
                    self.success_count = 0
            else:
                self.total_failures += 1
                self.success_count = 0  # Reset consecutive success count
                # Decrease concurrency immediately on failure
                self._decrease_limit()
    
    def _increase_limit(self):
        """Increase concurrency limit"""
        old_limit = self.current_limit
        new_limit = min(self.current_limit + self.increase_step, self.max_concurrent)
        
        if new_limit > old_limit:
            # Increase semaphore capacity
            for _ in range(new_limit - old_limit):
                self._semaphore.release()
            self.current_limit = new_limit
            logger.info(f"📈 Concurrency increased: {old_limit} → {new_limit} (consecutive successes: {self.success_threshold})")
    
    def _decrease_limit(self):
        """Decrease concurrency limit"""
        old_limit = self.current_limit
        new_limit = max(int(self.current_limit * self.decrease_ratio), self.min_concurrent)
        
        if new_limit < old_limit:
            # Decrease semaphore capacity (by acquiring)
            decrease_count = old_limit - new_limit
            acquired = 0
            for _ in range(decrease_count):
                if self._semaphore.acquire(blocking=False):
                    acquired += 1
            self.current_limit = old_limit - acquired  # Actual amount decreased
            logger.warning(f"📉 Concurrency decreased: {old_limit} → {self.current_limit} (failure/timeout detected)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "current_limit": self.current_limit,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "failure_rate": self.total_failures / max(1, self.total_requests),
            "success_streak": self.success_count
        }


# Global adaptive limiter instance
_adaptive_limiter = None

def get_adaptive_limiter() -> AdaptiveConcurrencyLimiter:
    """Get global adaptive concurrency limiter"""
    global _adaptive_limiter
    if _adaptive_limiter is None:
        _adaptive_limiter = AdaptiveConcurrencyLimiter()
    return _adaptive_limiter


class OpenAICompatibleLLM(BaseLLM):
    """Access vLLM service through OpenAI compatible /v1/chat/completions interface"""

    def __init__(self, config: Dict[str, Any]):
        # Required config keys: server_url, served_model_name, optional api_key
        self.server_url: str = config.get("server_url", "").rstrip("/")
        self.served_model_name: str = config.get("served_model_name", "")
        self.api_key: Optional[str] = config.get("api_key")  # Optional
        self.timeout: int = config.get("timeout", 120)
        if not self.server_url or not self.served_model_name:
            raise ValueError("OpenAICompatibleLLM requires server_url and served_model_name configuration")

        super().__init__(config)
        self.session = None
        self._load_model()

    def _load_model(self):
        # Create high-concurrency optimized Session
        self.session = requests.Session()
        
        # Configure connection pool - supports high concurrency
        adapter = HTTPAdapter(
            pool_connections=50,  # Connection pool size
            pool_maxsize=100,     # Maximum connections
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504]
            ),
            pool_block=False      # Non-blocking mode
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        # logger.info(f"Using OpenAI-compatible vLLM server: {self.server_url} model={self.served_model_name}")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure path doesn't duplicate /v1
        if self.server_url.endswith('/v1') and path.startswith('/v1/'):
            path = path[3:]  # Remove leading /v1
        url = f"{self.server_url}{path}"
        
        # 🎛️ Get adaptive concurrency limiter
        limiter = get_adaptive_limiter()
        
        # Add timeout retry mechanism, max 4 attempts
        max_retries = 4
        last_exception = None
        success = False
        
        # Wait to acquire execution permit
        if not limiter.acquire(timeout=300):
            raise requests.exceptions.Timeout("Timeout waiting for concurrency slot")
        
        try:
            for attempt in range(max_retries):
                try:
                    resp = self.session.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
                    resp.raise_for_status()
                    success = True
                    return resp.json()
                except requests.exceptions.Timeout as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                        logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Request timeout, max retries reached ({max_retries})")
                except requests.exceptions.ConnectionError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Connection error, max retries reached ({max_retries})")
            
            # All retries failed
            raise last_exception
        finally:
            # Release concurrency slot and report success/failure status
            limiter.release(success=success)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        logprobs: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Wrap single-turn prompt as chat-completions call"""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        # repetition_penalty is usually not in OpenAI chat schema, skip or pass to vLLM compatible params

        # For fine-tuned Alpaca format models, use completions API directly
        if "finetune" in self.served_model_name.lower() or "alpaca" in self.served_model_name.lower():
            # Use completions API directly
            completions_payload = {
                "model": self.served_model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if repetition_penalty is not None:
                completions_payload["repetition_penalty"] = repetition_penalty
            if logprobs is not None:
                completions_payload["logprobs"] = logprobs
            completions_payload.update({k: v for k, v in kwargs.items() if v is not None})
            
            data = self._post("/v1/completions", completions_payload)
            try:
                # If logprobs requested, return full response, otherwise return text only
                if logprobs is not None:
                    return data  # Return full response with logprobs
                return data["choices"][0]["text"].strip()
            except Exception:
                logger.error(f"Unexpected response from server: {data}")
                return ""
        
        # For other models, try chat completions
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.served_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if logprobs is not None:
            payload["logprobs"] = True  # Chat API uses boolean
            payload["top_logprobs"] = logprobs
        # Allow passing through other sampling parameters
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        # Try chat completions, fallback to completions API if failed
        try:
            data = self._post("/v1/chat/completions", payload)
            # If logprobs requested, return full response, otherwise return text only
            if logprobs is not None:
                logger.info(f"Returning full response with logprobs: {type(data)}")
                return data  # Return full response with logprobs
            return data["choices"][0]["message"]["content"].strip()
        except Exception as chat_error:
            logger.warning(f"Chat completions failed: {chat_error}, trying completions API")
            # Fallback to completions API
            completions_payload = {
                "model": self.served_model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if repetition_penalty is not None:
                completions_payload["repetition_penalty"] = repetition_penalty
            if logprobs is not None:
                completions_payload["logprobs"] = logprobs
            completions_payload.update({k: v for k, v in kwargs.items() if v is not None})
            
            data = self._post("/v1/completions", completions_payload)
            try:
                # If logprobs requested, return full response, otherwise return text only
                if logprobs is not None:
                    return data  # Return full response with logprobs
                return data["choices"][0]["text"].strip()
            except Exception:
                logger.error(f"Unexpected response from server: {data}")
                return ""

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        # Simple serial calls, maintains interface consistency (can be optimized to parallel if needed)
        results: List[str] = []
        for p in prompts:
            results.append(self.generate(p, **kwargs))
        return results

    def chat(
        self,
        messages: List[Dict[str, str]],
        chat_templates: Dict[str, Dict[str, str]],
        **kwargs,
    ) -> str:
        """Directly forward messages to /v1/chat/completions, without using template formatting"""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        repetition_penalty = kwargs.get("repetition_penalty")

        payload = {
            "model": self.served_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        # Allow passing through other parameters
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        data = self._post("/v1/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.error(f"Unexpected response from server: {data}")
            return ""
