"""Read-only suffix tracing for full-prompt Llama/Qwen mechanism replays.

Projection traces are projection outputs, NOT post-RoPE cache donors. Only
prefill is traced. Hash equality does not cover later generation activations.
"""


class SuffixTrace:
    def __init__(self, model, n, tau, chunk, tensor_hash):
        self.n, self.tau, self.chunk = int(n), int(tau), int(chunk)
        if self.chunk <= 0 or not 0 <= self.tau < self.n // self.chunk:
            raise ValueError("trace requires a complete suffix chunk")
        self.prefix = self.tau * self.chunk
        self.tensor_hash = tensor_hash
        self.records = {}
        self._handles = []
        self._regions = {"first_suffix_chunk": (self.prefix, self.prefix + self.chunk)}
        if self.n % self.chunk:
            self._regions["remainder"] = (self.n // self.chunk * self.chunk, self.n)
        try:
            for i, layer in enumerate(model.model.layers):
                name = "layer.%d" % i
                self._pre(layer.self_attn, name + ".attention_input")
                self._post(layer.self_attn, name + ".attention_output")
                for proj in ("q_proj", "k_proj", "v_proj"):
                    self._post(getattr(layer.self_attn, proj), name + "." + proj)
                self._pre(layer.mlp, name + ".mlp_input")
                self._post(layer.mlp, name + ".mlp_output")
                self._post(layer, name + ".residual")
            self._handles.append(model.model.norm.register_forward_hook(self._norm))
            self._handles.append(model.register_forward_hook(self._logits, with_kwargs=True))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _tensor(value):
        if hasattr(value, "shape"):
            return value
        if isinstance(value, (tuple, list)) and value:
            return SuffixTrace._tensor(value[0])
        return None

    @staticmethod
    def _hidden(args, kwargs):
        value = kwargs.get("hidden_states")
        return SuffixTrace._tensor(args[0] if value is None and args else value)

    def _save(self, key, tensor, region=None):
        if key in self.records:
            raise RuntimeError("duplicate prefill trace field: " + key)
        rec = {"sha256": self.tensor_hash(tensor), "shape": list(tensor.shape),
               "dtype": str(tensor.dtype)}
        if region is not None:
            rec["token_start"], rec["token_end"] = region
        self.records[key] = rec

    def _slices(self, name, tensor):
        if tensor is None or len(tensor.shape) < 3 or tensor.shape[1] != self.n:
            return
        for region, (a, b) in self._regions.items():
            self._save(name + "." + region, tensor[:, a:b, ...], (a, b))

    def _pre(self, module, name):
        def hook(module, args, kwargs):
            self._slices(name, self._hidden(args, kwargs))
        self._handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def _post(self, module, name):
        def hook(module, args, output):
            self._slices(name, self._tensor(output))
        self._handles.append(module.register_forward_hook(hook))

    def _norm(self, module, args, output):
        tensor = self._tensor(output)
        if tensor is not None and len(tensor.shape) >= 3 and tensor.shape[1] == self.n:
            self._save("final_norm.entire_suffix", tensor[:, self.prefix:, ...],
                       (self.prefix, self.n))

    def _logits(self, module, args, kwargs, output):
        # Generation may return only the final prefill logit. Determine whether
        # this is prefill from INPUT length, and hash exactly the returned logits.
        inputs = kwargs.get("input_ids")
        if inputs is None:
            inputs = kwargs.get("inputs_embeds")
        if inputs is None and args:
            inputs = args[0]
        if inputs is None or len(inputs.shape) < 2 or inputs.shape[1] != self.n:
            return
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, dict):
            logits = output.get("logits")
        if logits is None:
            logits = self._tensor(output)
        if logits is None:
            raise RuntimeError("prefill model output contains no traceable logits")
        self._save("model.first_forward_returned_logits", logits)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
