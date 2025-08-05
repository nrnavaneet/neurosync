"""
Connector for local Hugging Face Transformers models.
"""
import asyncio
from threading import Thread
from typing import Any, AsyncGenerator, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from neurosync.core.logging.logger import get_logger
from neurosync.serving.llm.base import BaseLLM

logger = get_logger(__name__)


class HuggingFaceLocalLLM(BaseLLM):
    """
    A connector for running local Hugging Face models for text generation.
    Supports streaming for real-time responses.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        logger.info(f"Loading local Hugging Face model: {model_name}")
        self.model_name = model_name

        # Quantization config for memory efficiency
        quantization_config = None
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Model loaded successfully.")

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generates a stream of text from the local model."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

        # Generation needs to be run in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
            await asyncio.sleep(0.01)  # Yield control to the event loop

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "huggingface_local",
            "model_name": self.model_name,
            "device": str(self.model.device),
        }
