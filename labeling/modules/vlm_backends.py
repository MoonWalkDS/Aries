"""Backend VLM thay được cho stage annotate.

Mọi backend nhận (danh sách ảnh PIL, prompt, system prompt) và trả văn bản thô + số liệu sử dụng.
Việc dựng prompt, parse JSON, validate và checkpoint nằm ở ``vlm_annotator.py`` và dùng chung.

- ``QwenLocalBackend``: Qwen2.5-VL qua Hugging Face transformers (Kaggle/Colab hoặc GPU riêng).
- ``ApiBackend``: Claude qua SDK ``anthropic``. Tốn phí theo token và gửi ảnh ra dịch vụ ngoài.
- ``OpenAIBackend``: GPT qua SDK ``openai``. Tốn phí theo token và gửi ảnh ra dịch vụ ngoài.
"""
from __future__ import annotations

import base64
import io
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

from .utils import free_gpu_memory, get_logger


class BackendError(RuntimeError):
    """Lỗi không khắc phục được bằng cách hỏi lại cùng nội dung (từ chối, hết quota, sai cấu hình)."""


@dataclass
class VLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class VLMBackend(ABC):
    """Giao diện chung của mọi backend."""

    name: str = "base"
    model_id: str = ""

    @abstractmethod
    def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> VLMResult:
        """Sinh văn bản từ ảnh + prompt. Phải trả về toàn bộ văn bản model sinh ra."""

    def close(self) -> None:
        """Giải phóng tài nguyên (bộ nhớ GPU, kết nối)."""


# --------------------------------------------------------------------------- Qwen local
class QwenLocalBackend(VLMBackend):
    """Qwen2.5-VL chạy tại chỗ. Đổi ``model_id`` trong config để dùng bản 3B/32B/72B."""

    name = "qwen_local"

    def __init__(self, cfg: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.log = get_logger()
        self._torch = torch
        qcfg = {**cfg["vlm"]["qwen_local"], **(overrides or {})}
        self.model_id = qcfg["model_id"]
        cache_dir = cfg["paths"].get("model_cache")

        dtype = torch.bfloat16 if qcfg.get("dtype") == "bfloat16" else torch.float16
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto", "cache_dir": cache_dir}
        if qcfg.get("quantization") == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        self.log.info("Nạp %s (dtype=%s, quantization=%s)", self.model_id, qcfg.get("dtype"), qcfg.get("quantization"))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs).eval()
        # min/max_pixels giới hạn số token ảnh, tức là giới hạn VRAM và thời gian mỗi ảnh.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=int(qcfg["min_pixels"]),
            max_pixels=int(qcfg["max_pixels"]),
            cache_dir=cache_dir,
        )

    def _generate_once(self, images: Sequence[Image.Image], prompt: str, system: Optional[str], max_new_tokens: int) -> VLMResult:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})

        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=list(images), padding=True, return_tensors="pt").to(self.model.device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        n_input = int(inputs["input_ids"].shape[1])
        new_tokens = generated[:, n_input:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return VLMResult(
            text=text,
            input_tokens=n_input,
            output_tokens=int(new_tokens.shape[1]),
            seconds=time.perf_counter() - started,
        )

    def generate(self, images: Sequence[Image.Image], prompt: str, system: Optional[str] = None, max_new_tokens: int = 1024) -> VLMResult:
        try:
            return self._generate_once(images, prompt, system, max_new_tokens)
        except self._torch.cuda.OutOfMemoryError:
            # Hết VRAM: dọn bộ nhớ, thu nhỏ ảnh còn 70% mỗi cạnh rồi thử lại đúng một lần.
            self.log.warning("CUDA hết bộ nhớ, thu nhỏ ảnh và thử lại")
            free_gpu_memory()
            smaller = [im.resize((max(28, int(im.width * 0.7)), max(28, int(im.height * 0.7))), Image.LANCZOS) for im in images]
            return self._generate_once(smaller, prompt, system, max_new_tokens)

    def close(self) -> None:
        self.model = None  # type: ignore[assignment]
        free_gpu_memory()


# --------------------------------------------------------------------------- Claude API
def _image_block(image: Image.Image) -> dict[str, Any]:
    """Mã hoá ảnh PIL thành khối ảnh base64 JPEG của Messages API."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


class ApiBackend(VLMBackend):
    """Claude qua Messages API.

    Khoá lấy theo cơ chế mặc định của SDK (``ANTHROPIC_API_KEY`` hoặc hồ sơ ``ant auth login``);
    file ``.env`` ở thư mục gốc dự án được nạp trước nếu có.
    Thinking luôn bật trên model mặc định nên ``max_tokens`` phải đủ cho cả phần suy luận.
    """

    name = "api"

    def __init__(self, cfg: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> None:
        import anthropic

        try:
            from dotenv import load_dotenv

            load_dotenv(Path(cfg["_config_dir"]).parent / ".env")
        except ImportError:
            pass

        acfg = {**cfg["vlm"]["api"], **(overrides or {})}
        if acfg.get("provider", "anthropic") != "anthropic":
            raise BackendError(f"Chưa hỗ trợ provider {acfg['provider']!r}; hiện chỉ có 'anthropic'.")
        self.log = get_logger()
        self._anthropic = anthropic
        self.model_id = acfg["model_id"]
        self.effort = acfg.get("effort", "medium")
        self.max_tokens = int(acfg.get("max_tokens", 16000))
        self.refusal_fallback = bool(acfg.get("refusal_fallback", True))
        self.client = anthropic.Anthropic(
            max_retries=int(acfg.get("max_retries", 4)),  # SDK tự retry 429/5xx/lỗi mạng có backoff
            timeout=float(acfg.get("timeout_s", 300)),
        )
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            self.log.warning("Chưa thấy ANTHROPIC_API_KEY; SDK sẽ thử hồ sơ đăng nhập `ant auth login`.")

    def generate(self, images: Sequence[Image.Image], prompt: str, system: Optional[str] = None, max_new_tokens: int = 1024) -> VLMResult:
        anthropic = self._anthropic
        content: list[dict[str, Any]] = [_image_block(im) for im in images]
        content.append({"type": "text", "text": prompt})
        request: dict[str, Any] = {
            "model": self.model_id,
            # max_new_tokens là phần trả lời; cộng thêm chỗ cho phần suy luận của model.
            "max_tokens": max(self.max_tokens, max_new_tokens),
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            request["system"] = system

        started = time.perf_counter()
        try:
            if self.refusal_fallback:
                # Khi model từ chối, server tự chạy lại cùng yêu cầu trên model dự phòng.
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
                )
            else:
                response = self.client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise BackendError("Khoá API không hợp lệ hoặc chưa cấu hình (ANTHROPIC_API_KEY).") from exc
        except anthropic.PermissionDeniedError as exc:
            raise BackendError(f"Khoá API không có quyền dùng {self.model_id}: {exc.message}") from exc
        except anthropic.BadRequestError as exc:
            raise BackendError(f"Yêu cầu không hợp lệ: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise BackendError("Vẫn bị giới hạn tốc độ sau khi SDK đã thử lại.") from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(f"API trả lỗi {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise BackendError(f"Không kết nối được tới API: {exc}") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise BackendError(f"Model từ chối yêu cầu (category={getattr(details, 'category', None)}).")
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "max_tokens":
            self.log.warning("Trả lời bị cắt vì chạm max_tokens=%d", request["max_tokens"])
        return VLMResult(
            text=text,
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
            seconds=time.perf_counter() - started,
        )


# --------------------------------------------------------------------------- OpenAI API
def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buffer.getvalue()).decode("ascii")


class OpenAIBackend(VLMBackend):
    """Model GPT qua Chat Completions API của SDK ``openai``.

    Khoá lấy từ ``OPENAI_API_KEY``; file ``.env`` ở thư mục gốc dự án được nạp trước nếu có.
    Tốn phí theo token và gửi ảnh ra dịch vụ ngoài. Chấp nhận lời gọi không có ảnh (thuần văn bản).
    """

    name = "openai"

    def __init__(self, cfg: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> None:
        import openai

        try:
            from dotenv import load_dotenv

            load_dotenv(Path(cfg["_config_dir"]).parent / ".env")
        except ImportError:
            pass

        ocfg = {**cfg["vlm"]["openai"], **(overrides or {})}
        if not os.getenv("OPENAI_API_KEY"):
            raise BackendError("Chưa có OPENAI_API_KEY. Đặt trong file .env ở thư mục gốc dự án hoặc trong biến môi trường.")
        self.log = get_logger()
        self._openai = openai
        self.model_id = ocfg["model_id"]
        self.reasoning_effort = ocfg.get("reasoning_effort")
        self.max_tokens = int(ocfg.get("max_tokens", 8000))
        self.image_detail = ocfg.get("image_detail", "auto")
        self.client = openai.OpenAI(
            max_retries=int(ocfg.get("max_retries", 4)),  # SDK tự retry 429/5xx/lỗi mạng có backoff
            timeout=float(ocfg.get("timeout_s", 300)),
        )

    def generate(self, images: Sequence[Image.Image], prompt: str, system: Optional[str] = None, max_new_tokens: int = 1024) -> VLMResult:
        openai = self._openai
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": _data_url(im), "detail": self.image_detail}} for im in images]
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            # Với model có suy luận, giới hạn này gồm cả token suy luận nên phải rộng hơn phần trả lời.
            "max_completion_tokens": max(self.max_tokens, max_new_tokens),
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**request)
        except openai.AuthenticationError as exc:
            raise BackendError("Khoá OpenAI không hợp lệ (OPENAI_API_KEY).") from exc
        except openai.PermissionDeniedError as exc:
            raise BackendError(f"Khoá OpenAI không có quyền dùng {self.model_id}: {exc.message}") from exc
        except openai.NotFoundError as exc:
            raise BackendError(f"Không có model {self.model_id!r}; sửa vlm.openai.model_id. {exc.message}") from exc
        except openai.BadRequestError as exc:
            raise BackendError(f"Yêu cầu không hợp lệ: {exc.message}") from exc
        except openai.RateLimitError as exc:
            raise BackendError("Bị giới hạn tốc độ hoặc hết hạn mức sau khi SDK đã thử lại.") from exc
        except openai.APIStatusError as exc:
            raise BackendError(f"API trả lỗi {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise BackendError(f"Không kết nối được tới API: {exc}") from exc

        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise BackendError(f"Model từ chối yêu cầu: {choice.message.refusal}")
        if choice.finish_reason == "content_filter":
            raise BackendError("Câu trả lời bị bộ lọc nội dung chặn.")
        if choice.finish_reason == "length":
            self.log.warning("Trả lời bị cắt vì chạm max_completion_tokens=%d", request["max_completion_tokens"])
        usage = response.usage
        return VLMResult(
            text=choice.message.content or "",
            input_tokens=int(usage.prompt_tokens) if usage else 0,
            output_tokens=int(usage.completion_tokens) if usage else 0,
            seconds=time.perf_counter() - started,
        )


# --------------------------------------------------------------------------- factory
def create_backend(cfg: dict[str, Any], backend: Optional[str] = None, overrides: Optional[dict[str, Any]] = None) -> VLMBackend:
    """Tạo backend theo ``vlm.backend`` trong config (hoặc tên truyền vào)."""
    name = backend or cfg["vlm"]["backend"]
    if name == "qwen_local":
        return QwenLocalBackend(cfg, overrides)
    if name == "api":
        return ApiBackend(cfg, overrides)
    if name == "openai":
        return OpenAIBackend(cfg, overrides)
    raise ValueError(f"vlm.backend không hợp lệ: {name!r} (chọn qwen_local, api hoặc openai)")
