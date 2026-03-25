from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **gen_kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class HFModelBackend(ModelBackend):
    """
    Generic HuggingFace causal LM backend.
    """

    def __init__(self, model_name: str, trust_remote_code: bool = True, cache_dir: str | None = None):
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        # Silence repeated Transformers warning:
        # "Setting `pad_token_id` to `eos_token_id` ... for open-end generation."
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_params: Dict[str, Any] = {
            "temperature": gen_kwargs.get("temperature", 0.2),
            "top_p": gen_kwargs.get("top_p", 1.0),
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 512),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        outputs = self.model.generate(**inputs, **gen_params)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def name(self) -> str:
        return self._model_name


class QwenMathBackend(HFModelBackend):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-1.5B", cache_dir: str | None = None):
        super().__init__(model_name=model_name, trust_remote_code=True, cache_dir=cache_dir)


class DeepSeekR1Backend(HFModelBackend):
    """
    Backend for DeepSeek-R1-Distill-Qwen-1.5B.
    """

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir: str | None = None):
        super().__init__(model_name=model_name, trust_remote_code=True, cache_dir=cache_dir)

