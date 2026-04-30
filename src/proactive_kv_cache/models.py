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
    used_past_key_values: bool = False
    cache_prepare_latency_ms: float = 0.0
    cache_fallback_reason: str | None = None
    prepared_past_length: int = 0


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

    def logit_guard_distance(self, prefix_a: Sequence[int], prefix_b: Sequence[int], top_k: int = 32) -> float | None:
        """Return a small distance when two prefixes induce similar next-token logits.

        Backends that cannot expose logits return ``None``. ShadowKV++ uses this
        for the guarded semantic-reuse ablation rather than the default safe path.
        """
        return None

    def prepare_past_key_values(self, past_key_values: Any) -> Any:
        return past_key_values

    def move_kv_cache(self, past_key_values: Any, target: str) -> Any:
        return past_key_values

    def estimate_prefill_cost_ms(self, token_count: int) -> float:
        return float(max(token_count, 0))

    def estimate_kv_cache_bytes(self, token_count: int) -> int:
        return int(max(token_count, 0) * 1024)


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
        return tuple(tokens)

    def decode(self, tokens: Sequence[int]) -> str:
        return ' '.join(self._inverse_vocab.get(int(t), str(t)) for t in tokens)

    def logit_guard_distance(self, prefix_a: Sequence[int], prefix_b: Sequence[int], top_k: int = 32) -> float | None:
        # Deterministic lightweight proxy: Jaccard distance over recent tokens.
        a = set(tuple(prefix_a)[-max(top_k, 1):])
        b = set(tuple(prefix_b)[-max(top_k, 1):])
        if not a and not b:
            return 0.0
        return 1.0 - (len(a & b) / max(len(a | b), 1))

    def estimate_prefill_cost_ms(self, token_count: int) -> float:
        latency_per_token = 2.5 if self.device == 'cpu' else 1.0
        return float(latency_per_token * max(token_count, 0) + 0.25)

    def estimate_kv_cache_bytes(self, token_count: int) -> int:
        return int(max(token_count, 0) * (128 if self.device == 'cpu' else 160))

    def prefill(self, tokens: Sequence[int], past_key_values: Any = None) -> PrefillResult:
        token_count = len(tokens)
        latency_ms = self.estimate_prefill_cost_ms(token_count)
        past_tokens: Tuple[int, ...] = ()
        if isinstance(past_key_values, dict):
            past_tokens = tuple(past_key_values.get('tokens', ()))
            if not past_tokens and int(past_key_values.get('prefix_len', 0)) > 0:
                # Older tests and lightweight simulations sometimes store only
                # a prefix length placeholder. Treat that as a valid prepared
                # cache for latency/accounting purposes.
                past_tokens = tuple([0] * int(past_key_values.get('prefix_len', 0)))
        full_tokens = past_tokens + tuple(tokens)
        memory_bytes = int(max(len(full_tokens), 1) * (128 if self.device == 'cpu' else 160))
        kv = {'prefix_len': len(full_tokens), 'tokens': full_tokens, 'device': self.device}
        time.sleep(min(latency_ms / 1000.0, 0.01))
        gpu_util = 35.0 if self.device.startswith('cuda') else None
        return PrefillResult(
            kv_cache=kv,
            latency_ms=latency_ms,
            memory_bytes=memory_bytes,
            device=self.device,
            gpu_utilization_pct=gpu_util,
            used_past_key_values=bool(past_tokens),
            prepared_past_length=len(past_tokens),
        )

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
        try:
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
                load_kwargs['torch_dtype'] = model_dtype
            if device.startswith('cuda'):
                load_kwargs['device_map'] = device
                load_kwargs['low_cpu_mem_usage'] = True
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            self.model.eval()
            if not device.startswith('cuda'):
                self.model.to(device)
        except Exception as exc:
            raise RuntimeError(
                f'Failed to load Hugging Face model {model_name!r} on device {device!r}. '
                'Check model access, local dependencies, and device availability.'
            ) from exc
        self._nvml = _try_init_nvml(device)
        self.max_positions = int(getattr(self.model.config, 'n_positions', None) or getattr(self.model.config, 'max_position_embeddings', None) or 1024)
        config = self.model.config
        num_layers = int(getattr(config, 'num_hidden_layers', None) or getattr(config, 'n_layer', None) or 0)
        num_heads = int(getattr(config, 'num_attention_heads', None) or getattr(config, 'n_head', None) or 0)
        num_kv_heads = int(getattr(config, 'num_key_value_heads', None) or max(num_heads, 1))
        hidden_size = int(getattr(config, 'hidden_size', None) or getattr(config, 'n_embd', None) or 0)
        head_dim = int(getattr(config, 'head_dim', None) or (hidden_size // max(num_heads, 1) if hidden_size and num_heads else 0) or 64)
        try:
            dtype_bytes = int(next(self.model.parameters()).element_size())
        except Exception:
            dtype_bytes = 2 if device.startswith('cuda') else 4
        self._kv_bytes_per_token = max(2 * max(num_layers, 1) * max(num_kv_heads, 1) * max(head_dim, 1) * max(dtype_bytes, 1), 1)

    def tokenize(self, text: str) -> Tuple[int, ...]:
        encoded = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_positions)
        return tuple(encoded['input_ids'][0].tolist())

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens))

    def logit_guard_distance(self, prefix_a: Sequence[int], prefix_b: Sequence[int], top_k: int = 32) -> float | None:
        """Compare next-token distributions after two candidate prefixes.

        Returns symmetric total-variation distance over the union of top-k
        token ids from both next-token distributions. Lower is safer.
        """
        if not prefix_a or not prefix_b:
            return None
        max_len = max(min(len(prefix_a), len(prefix_b), self.max_positions), 1)
        a = list(prefix_a)[-max_len:]
        b = list(prefix_b)[-max_len:]
        try:
            with self.torch.no_grad():
                ids_a = self.torch.tensor([a], dtype=self.torch.long, device=self.device)
                ids_b = self.torch.tensor([b], dtype=self.torch.long, device=self.device)
                logits_a = self.model(input_ids=ids_a, use_cache=False).logits[0, -1].float()
                logits_b = self.model(input_ids=ids_b, use_cache=False).logits[0, -1].float()
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize()
                k = int(max(min(top_k, logits_a.numel(), logits_b.numel()), 1))
                top_a = self.torch.topk(logits_a, k).indices
                top_b = self.torch.topk(logits_b, k).indices
                union = self.torch.unique(self.torch.cat([top_a, top_b]))
                pa = self.torch.softmax(logits_a[union], dim=-1)
                pb = self.torch.softmax(logits_b[union], dim=-1)
                return float(0.5 * self.torch.sum(self.torch.abs(pa - pb)).item())
        except Exception:
            return None

    def estimate_prefill_cost_ms(self, token_count: int) -> float:
        if self.device.startswith('cuda'):
            return float(0.6 * max(token_count, 0) + 5.0)
        return float(1.1 * max(token_count, 0) + 8.0)

    def prepare_past_key_values(self, past_key_values: Any) -> Any:
        moved = self.move_kv_cache(past_key_values, self.device)
        trimmed = self._trim_past_key_values(moved, self.max_positions)
        return self._normalize_past_key_values(trimmed)

    def _normalize_past_key_values(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values
        if isinstance(past_key_values, tuple):
            from transformers.cache_utils import DynamicCache
            return DynamicCache.from_legacy_cache(past_key_values)
        return past_key_values

    def move_kv_cache(self, past_key_values: Any, target: str) -> Any:
        if past_key_values is None:
            return None
        moved_layers = []
        try:
            for layer in past_key_values:
                moved_layer = []
                for t in layer:
                    if t is None:
                        moved_layer.append(None)
                    elif hasattr(t, 'to'):
                        moved_layer.append(t.to(target))
                    else:
                        moved_layer.append(t)
                moved_layers.append(tuple(moved_layer))
            return tuple(moved_layers)
        except Exception as exc:
            raise RuntimeError(f'Failed to move past_key_values to {target!r} for model {self.model_name!r}') from exc

    def estimate_kv_cache_bytes(self, token_count: int) -> int:
        return int(max(token_count, 0) * self._kv_bytes_per_token)

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
        prepare_start = time.perf_counter()
        prepared_kv = self.prepare_past_key_values(past_key_values)
        cache_prepare_latency_ms = (time.perf_counter() - prepare_start) * 1000.0

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

        prepared_past_length = self._past_length(prepared_kv)
        if len(tokens) == 0:
            return PrefillResult(
                kv_cache=prepared_kv,
                latency_ms=cache_prepare_latency_ms,
                memory_bytes=estimate_past_key_values_bytes(prepared_kv),
                device=self.device,
                gpu_utilization_pct=None,
                used_past_key_values=prepared_past_length > 0,
                cache_prepare_latency_ms=cache_prepare_latency_ms,
                prepared_past_length=prepared_past_length,
            )

        input_ids = self.torch.tensor([tokens], dtype=self.torch.long, device=self.device)
        position_ids = self.torch.arange(past_len, past_len + len(tokens), dtype=self.torch.long, device=self.device).unsqueeze(0)
        attention_mask = self.torch.ones((1, past_len + len(tokens)), dtype=self.torch.long, device=self.device)

        util_before = _read_gpu_utilization(self._nvml)
        start = time.perf_counter()
        cache_fallback_reason = None
        used_past_key_values = prepared_past_length > 0
        try:
            with self.torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    past_key_values=prepared_kv,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
        except (IndexError, RuntimeError, ValueError) as exc:
            cache_fallback_reason = f'backend_retry_without_cache:{exc.__class__.__name__}'
            used_past_key_values = False
            prepared_kv = None
            input_ids = self.torch.tensor([tokens[-self.max_positions:]], dtype=self.torch.long, device=self.device)
            position_ids = self.torch.arange(0, input_ids.shape[-1], dtype=self.torch.long, device=self.device).unsqueeze(0)
            attention_mask = self.torch.ones((1, input_ids.shape[-1]), dtype=self.torch.long, device=self.device)
            with self.torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    past_key_values=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        latency_ms = cache_prepare_latency_ms + (time.perf_counter() - start) * 1000.0
        util_after = _read_gpu_utilization(self._nvml)
        util = None
        if util_before is not None and util_after is not None:
            util = float((util_before + util_after) / 2.0)
        normalized_output_kv = self._normalize_past_key_values(output.past_key_values)
        memory_bytes = estimate_past_key_values_bytes(normalized_output_kv)
        return PrefillResult(
            kv_cache=normalized_output_kv,
            latency_ms=latency_ms,
            memory_bytes=memory_bytes,
            device=self.device,
            gpu_utilization_pct=util,
            used_past_key_values=used_past_key_values,
            cache_prepare_latency_ms=cache_prepare_latency_ms,
            cache_fallback_reason=cache_fallback_reason,
            prepared_past_length=prepared_past_length if used_past_key_values else 0,
        )


class VLLMBackend(Backend):
    backend_name = 'vllm'
    default_min_reuse_prefix_tokens = 48
    default_min_store_prefix_tokens = 32
    default_cache_reuse_overhead_ms = 0.0
    supports_external_kv = False
    supports_native_prefix_caching = True

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda:0',
        dtype: str = 'auto',
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
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
            'trust_remote_code': trust_remote_code,
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

    def logit_guard_distance(self, prefix_a: Sequence[int], prefix_b: Sequence[int], top_k: int = 32) -> float | None:
        """Compare next-token distributions after two candidate prefixes.

        Returns symmetric total-variation distance over the union of top-k
        token ids from both next-token distributions. Lower is safer.
        """
        if not prefix_a or not prefix_b:
            return None
        max_len = max(min(len(prefix_a), len(prefix_b), self.max_positions), 1)
        a = list(prefix_a)[-max_len:]
        b = list(prefix_b)[-max_len:]
        try:
            with self.torch.no_grad():
                ids_a = self.torch.tensor([a], dtype=self.torch.long, device=self.device)
                ids_b = self.torch.tensor([b], dtype=self.torch.long, device=self.device)
                logits_a = self.model(input_ids=ids_a, use_cache=False).logits[0, -1].float()
                logits_b = self.model(input_ids=ids_b, use_cache=False).logits[0, -1].float()
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize()
                k = int(max(min(top_k, logits_a.numel(), logits_b.numel()), 1))
                top_a = self.torch.topk(logits_a, k).indices
                top_b = self.torch.topk(logits_b, k).indices
                union = self.torch.unique(self.torch.cat([top_a, top_b]))
                pa = self.torch.softmax(logits_a[union], dim=-1)
                pb = self.torch.softmax(logits_b[union], dim=-1)
                return float(0.5 * self.torch.sum(self.torch.abs(pa - pb)).item())
        except Exception:
            return None

    def estimate_prefill_cost_ms(self, token_count: int) -> float:
        return float(0.35 * max(token_count, 0) + 4.0)

    def estimate_kv_cache_bytes(self, token_count: int) -> int:
        return int(max(token_count, 0) * 256)

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
        return PrefillResult(
            kv_cache=meta,
            latency_ms=latency_ms,
            memory_bytes=memory_bytes,
            device=self.device,
            gpu_utilization_pct=util,
            used_past_key_values=False,
        )


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


def load_backend(
    backend: str,
    model_name: str | None = None,
    device: str = 'cpu',
    dtype: str = 'auto',
    tensor_parallel_size: int = 1,
    enable_prefix_caching: bool = True,
    trust_remote_code: bool = False,
) -> Backend:
    if backend == 'fake':
        return FakeBackend(device=device)
    if backend == 'hf':
        if not model_name:
            raise ValueError('model_name is required for hf backend')
        return HuggingFaceBackend(model_name=model_name, device=device, dtype=dtype)
    if backend == 'vllm':
        if not model_name:
            raise ValueError('model_name is required for vllm backend')
        return VLLMBackend(
            model_name=model_name,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=trust_remote_code,
        )
    raise ValueError(f'Unknown backend: {backend}')
