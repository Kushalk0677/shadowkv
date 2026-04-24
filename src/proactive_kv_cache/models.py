from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class PrefillResult:
    kv_cache: Any
    latency_ms: float
    memory_bytes: int
    device: str
    gpu_utilization_pct: float | None = None


class Backend:
    device: str = 'cpu'
    backend_name: str = 'base'
    default_min_reuse_prefix_tokens: int = 8
    default_min_store_prefix_tokens: int = 8
    default_cache_reuse_overhead_ms: float = 2.0
    supports_external_kv: bool = True
    supports_native_prefix_caching: bool = False

    def tokenize(self, text: str) -> Tuple[int, ...]:
        raise NotImplementedError

    def decode(self, tokens: Sequence[int]) -> str:
        raise NotImplementedError

    def prefill(self, tokens: Sequence[int], past_key_values: Any = None) -> PrefillResult:
        raise NotImplementedError

    def prepare_past_key_values(self, past_key_values: Any) -> Any:
        return past_key_values

    def move_kv_cache(self, past_key_values: Any, target: str) -> Any:
        return past_key_values

    def estimate_prefill_cost_ms(self, token_count: int) -> float:
        return float(max(token_count, 0))


class FakeBackend(Backend):
    backend_name = 'fake'
    default_min_reuse_prefix_tokens = 3
    default_min_store_prefix_tokens = 3
    default_cache_reuse_overhead_ms = 0.25

    def __init__(self, device: str = 'cpu') -> None:
        self.device = device
        self._vocab: Dict[str, int] = {}
        self._inverse_vocab: Dict[int, str] = {}

    def tokenize(self, text: str) -> Tuple[int, ...]:
        tokens: List[int] = []
        for word in text.strip().split():
            if word not in self._vocab:
                token_id = len(self._vocab) + 1
                self._vocab[word] = token_id
                self._inverse_vocab[token_id] = word
            tokens.append(self._vocab[word])
        return tuple(tokens or [0])

    def decode(self, tokens: Sequence[int]) -> str:
        return ' '.join(self._inverse_vocab.get(int(t), str(t)) for t in tokens)

    def estimate_prefill_cost_ms(self, token_count: int) -> float:
        latency_per_token = 2.5 if self.device == 'cpu' else 1.0
        return float(latency_per_token * max(token_count, 0) + 0.25)

    def prefill(self, tokens: Sequence[int], past_key_values: Any = None) -> PrefillResult:
        token_count = len(tokens)
        latency_ms = self.estimate_prefill_cost_ms(token_count)
        past_tokens: Tuple[int, ...] = ()
        if isinstance(past_key_values, dict):
            past_tokens = tuple(past_key_values.get('tokens', ()))
        full_tokens = past_tokens + tuple(tokens)
        memory_bytes = int(max(len(full_tokens), 1) * (128 if self.device == 'cpu' else 160))
        kv = {'prefix_len': len(full_tokens), 'tokens': full_tokens, 'device': self.device}
        time.sleep(min(latency_ms / 1000.0, 0.01))
        gpu_util = 35.0 if self.device.startswith('cuda') else None
        return PrefillResult(kv_cache=kv, latency_ms=latency_ms, memory_bytes=memory_bytes, device=self.device, gpu_utilization_pct=gpu_util)

    def move_kv_cache(self, past_key_values: Any, target: str) -> Any:
        if isinstance(past_key_values, dict):
            moved = dict(past_key_values)
            moved['device'] = target
            return moved
        return past_key_values


class HuggingFaceBackend(Backend):
    backend_name = 'hf'
    default_min_reuse_prefix_tokens = 16
    default_min_store_prefix_tokens = 12
    default_cache_reuse_overhead_ms = 8.0
    supports_external_kv = True

    def __init__(self, model_name: str, device: str = 'cpu', dtype: str = 'auto') -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_dtype = None
        if dtype == 'float16':
            model_dtype = torch.float16
        elif dtype == 'bfloat16':
            model_dtype = torch.bfloat16
        elif device.startswith('cuda'):
            model_dtype = torch.float16

        load_kwargs = {}
        if model_dtype is not None:
            load_kwargs['dtype'] = model_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.model.to(device)
        self._nvml = _try_init_nvml(device)
        self.max_positions = int(getattr(self.model.config, 'n_positions', None) or getattr(self.model.config, 'max_position_embeddings', None) or 1024)

    def tokenize(self, text: str) -> Tuple[int, ...]:
        encoded = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_positions)
        return tuple(encoded['input_ids'][0].tolist())

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens))

    def estimate_prefill_cost_ms(self, token_count: int) -> float:
        if self.device.startswith('cuda'):
            return float(0.6 * max(token_count, 0) + 5.0)
        return float(1.1 * max(token_count, 0) + 8.0)

    def prepare_past_key_values(self, past_key_values: Any) -> Any:
        moved = self.move_kv_cache(past_key_values, self.device)
        return self._trim_past_key_values(moved, self.max_positions)

    def move_kv_cache(self, past_key_values: Any, target: str) -> Any:
        if past_key_values is None:
            return None
        moved_layers = []
        try:
            for layer in past_key_values:
                moved_layers.append(tuple(t.to(target) for t in layer))
            return tuple(moved_layers)
        except Exception:
            return past_key_values

    def _past_length(self, past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        try:
            first_layer = past_key_values[0]
            first_tensor = first_layer[0]
            return int(first_tensor.shape[-2])
        except Exception:
            return 0

    def _trim_past_key_values(self, past_key_values: Any, keep_last_tokens: int) -> Any:
        if past_key_values is None:
            return None
        if keep_last_tokens <= 0:
            return None
        try:
            trimmed_layers = []
            for layer in past_key_values:
                trimmed_layer = []
                for tensor in layer:
                    if tensor.shape[-2] > keep_last_tokens:
                        trimmed_tensor = tensor[..., -keep_last_tokens:, :]
                    else:
                        trimmed_tensor = tensor
                    trimmed_layer.append(trimmed_tensor)
                trimmed_layers.append(tuple(trimmed_layer))
            return tuple(trimmed_layers)
        except Exception:
            return past_key_values

    def prefill(self, tokens: Sequence[int], past_key_values: Any = None) -> PrefillResult:
        tokens = list(tokens)
        prepared_kv = self.prepare_past_key_values(past_key_values)

        if len(tokens) > self.max_positions:
            tokens = tokens[-self.max_positions:]

        past_len = self._past_length(prepared_kv)
        if past_len >= self.max_positions:
            prepared_kv = None
            past_len = 0

        total_len = past_len + len(tokens)
        if total_len > self.max_positions:
            allowed_tokens = self.max_positions - past_len
            if allowed_tokens <= 0:
                prepared_kv = None
                past_len = 0
                tokens = tokens[-self.max_positions:]
            else:
                tokens = tokens[-allowed_tokens:]

        past_len = self._past_length(prepared_kv)
        if past_len + len(tokens) > self.max_positions:
            keep_past = max(self.max_positions - len(tokens), 0)
            prepared_kv = self._trim_past_key_values(prepared_kv, keep_past)
            past_len = self._past_length(prepared_kv)

        if past_len + len(tokens) > self.max_positions:
            prepared_kv = None
            past_len = 0
            tokens = tokens[-self.max_positions:]

        if len(tokens) == 0:
            return PrefillResult(kv_cache=prepared_kv, latency_ms=0.0, memory_bytes=estimate_past_key_values_bytes(prepared_kv), device=self.device, gpu_utilization_pct=None)

        input_ids = self.torch.tensor([tokens], dtype=self.torch.long, device=self.device)
        position_ids = self.torch.arange(past_len, past_len + len(tokens), dtype=self.torch.long, device=self.device).unsqueeze(0)

        util_before = _read_gpu_utilization(self._nvml)
        start = time.perf_counter()
        try:
            with self.torch.no_grad():
                output = self.model(input_ids=input_ids, use_cache=True, past_key_values=prepared_kv, position_ids=position_ids)
        except (IndexError, RuntimeError, ValueError):
            prepared_kv = None
            input_ids = self.torch.tensor([tokens[-self.max_positions:]], dtype=self.torch.long, device=self.device)
            position_ids = self.torch.arange(0, input_ids.shape[-1], dtype=self.torch.long, device=self.device).unsqueeze(0)
            with self.torch.no_grad():
                output = self.model(input_ids=input_ids, use_cache=True, past_key_values=None, position_ids=position_ids)
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        util_after = _read_gpu_utilization(self._nvml)
        util = None
        if util_before is not None and util_after is not None:
            util = float((util_before + util_after) / 2.0)
        memory_bytes = estimate_past_key_values_bytes(output.past_key_values)
        return PrefillResult(kv_cache=output.past_key_values, latency_ms=latency_ms, memory_bytes=memory_bytes, device=self.device, gpu_utilization_pct=util)


class VLLMBackend(Backend):
    backend_name = 'vllm'
    default_min_reuse_prefix_tokens = 48
    default_min_store_prefix_tokens = 32
    default_cache_reuse_overhead_ms = 0.0
    supports_external_kv = False
    supports_native_prefix_caching = True

    def __init__(self, model_name: str, device: str = 'cuda:0', dtype: str = 'auto', tensor_parallel_size: int = 1, enable_prefix_caching: bool = True) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError('vLLM backend requires `pip install vllm` and a compatible CUDA environment.') from exc
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_prefix_caching = enable_prefix_caching
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        llm_kwargs = {
            'model': model_name,
            'tensor_parallel_size': tensor_parallel_size,
            'enable_prefix_caching': enable_prefix_caching,
            'trust_remote_code': True,
        }
        if dtype != 'auto':
            llm_kwargs['dtype'] = dtype
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        self.max_positions = 4096
        self._nvml = _try_init_nvml(device)

    def tokenize(self, text: str) -> Tuple[int, ...]:
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_positions)['input_ids']
        return tuple(encoded)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens))

    def estimate_prefill_cost_ms(self, token_count: int) -> float:
        return float(0.35 * max(token_count, 0) + 4.0)

    def prefill(self, tokens: Sequence[int], past_key_values: Any = None) -> PrefillResult:
        if past_key_values is not None:
            raise RuntimeError('vLLM backend does not expose external past_key_values for custom KV reuse. Use native_prefix_cache baseline instead.')
        prompt_token_ids = [list(tokens)]
        util_before = _read_gpu_utilization(self._nvml)
        start = time.perf_counter()
        outputs = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params, use_tqdm=False)
        latency_ms = (time.perf_counter() - start) * 1000.0
        util_after = _read_gpu_utilization(self._nvml)
        util = None
        if util_before is not None and util_after is not None:
            util = float((util_before + util_after) / 2.0)
        generated = outputs[0]
        token_count = len(tokens) + len(generated.outputs[0].token_ids)
        memory_bytes = int(max(token_count, 1) * 256)
        meta = {'native_prefix_caching': self.enable_prefix_caching, 'token_count': token_count}
        return PrefillResult(kv_cache=meta, latency_ms=latency_ms, memory_bytes=memory_bytes, device=self.device, gpu_utilization_pct=util)


def estimate_past_key_values_bytes(past_key_values: Any) -> int:
    total = 0
    try:
        for layer in past_key_values:
            for tensor in layer:
                total += tensor.numel() * tensor.element_size()
    except Exception:
        total = 0
    return int(total)


def _try_init_nvml(device: str) -> Any:
    if not device.startswith('cuda'):
        return None
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        index = 0
        if ':' in device:
            index = int(device.split(':', 1)[1])
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return pynvml, handle
    except Exception:
        return None


def _read_gpu_utilization(nvml_state: Any) -> float | None:
    if nvml_state is None:
        return None
    try:
        pynvml, handle = nvml_state
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        return None


def supports_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_backend(backend: str, model_name: str | None = None, device: str = 'cpu', dtype: str = 'auto', tensor_parallel_size: int = 1, enable_prefix_caching: bool = True) -> Backend:
    if backend == 'fake':
        return FakeBackend(device=device)
    if backend == 'hf':
        if not model_name:
            raise ValueError('model_name is required for hf backend')
        return HuggingFaceBackend(model_name=model_name, device=device, dtype=dtype)
    if backend == 'vllm':
        if not model_name:
            raise ValueError('model_name is required for vllm backend')
        return VLLMBackend(model_name=model_name, device=device, dtype=dtype, tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=enable_prefix_caching)
    raise ValueError(f'Unknown backend: {backend}')
