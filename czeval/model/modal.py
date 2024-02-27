import os
from typing import List

from modal import Image, Secret, Stub, build, enter, method
from modal.gpu import A100

image = Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
).pip_install(
    "huggingface_hub",
    "hf-transfer",
    "transformers",
    "torch==2.1.2",
    "jinja2==3.1.3",
    "accelerate",
    "tqdm==4.66.2",
)

stub = Stub(f"cz-eval", image=image)


@stub.cls(
    secrets=[Secret.from_name("my-huggingface-secret")],
    gpu=A100(memory=80),
    timeout=None,
)
class Model:
    def __init__(
        self,
        model_name: str = "name",
        chat_template: str | None = None,
        max_new_tokens=500,
        device="cpu",
    ) -> None:
        self.model_name = model_name
        self.chat_template = chat_template
        self.model_dir = "/modal"
        self.device = device
        self.max_new_tokens = max_new_tokens

    # @build()
    def download_model_and_load(self):
        if not self.model_name or not self.model_dir:
            return

        from huggingface_hub import snapshot_download
        from transformers.utils import move_cache

        os.makedirs(self.model_dir, exist_ok=True)

        snapshot_download(
            self.model_name,
            local_dir=self.model_dir,
            token=os.environ.get("HF_TOKEN"),
            ignore_patterns=["*.pt", "*.gguf"],
        )
        move_cache()

    @enter()
    def prepare(self):
        self.download_model_and_load()
        self.AUTO_MODEL_CLASS = self._get_model_type()
        self.model = self._get_model().to(self.device)

    def _get_model_type(self):
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        )

        config = AutoConfig.from_pretrained(self.model_dir)
        if config.model_type in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
            return AutoModelForSeq2SeqLM
        return AutoModelForCausalLM

    def _get_model(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

        if self.AUTO_MODEL_CLASS == AutoModelForSeq2SeqLM:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_dir, torch_dtype=torch.float16
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=torch.float16
        )

        model.to_bettertransformer()
        return model

    def _model_call(self, inps, attn_mask=None, labels=None):
        import torch
        import transformers

        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                return self.model(inps).logits

    def _clear_torch_cache(self) -> None:
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    def _get_prompts(self, tokenizer, turns: List[List[dict[str, str]]]):
        if self.chat_template:
            tokenizer.chat_template = self.chat_template
        if not tokenizer.chat_template:
            raise ValueError("Chat template is not set")
        return [
            tokenizer.apply_chat_template(
                turn, tokenize=False, add_generation_prompt=True
            )
            for turn in turns
        ]

    def _tokenize(self, tokenizer, prompts: List[str]):
        return tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

    def _detect_batch_size(self, tokenizer, prompts: List[str]):
        from accelerate import find_executable_batch_size
        from transformers import AutoModelForSeq2SeqLM
        import torch.nn.functional as F
        import torch

        _max_b_size = 100
        max_prefix_tokens = max(
            (
                self._tokenize(
                    tokenizer, prompts[batch_i : batch_i + _max_b_size]
                ).input_ids.size(1)
                for batch_i in range(0, len(prompts), _max_b_size)
            )
        )

        print(f"Max prefix tokens: {max_prefix_tokens}")

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=512)
        def forward_batch(batch_size):
            print(f"Trying batch size {batch_size}")
            if self.AUTO_MODEL_CLASS == AutoModelForSeq2SeqLM:
                batched_conts = torch.ones(
                    (batch_size, self.max_new_tokens), device=self.device
                ).long()
                test_batch = torch.ones(
                    (batch_size, max_prefix_tokens), device=self.device
                ).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, self.max_new_tokens + max_prefix_tokens),
                    device=self.device,
                ).long()

            for _ in range(5):
                _ = F.log_softmax(self._model_call(test_batch, **call_kwargs), dim=-1)  # noqa: F841

            return batch_size

        try:
            new_batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                new_batch_size = 1
            else:
                raise

            self._clear_torch_cache()
        return new_batch_size

    @method()
    def predict_basic(
        self,
        turns: List[List[dict[str, str]]],
        temperate: int,
        batch_size: int | None = None,
    ):
        from transformers import AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        prompts = self._get_prompts(tokenizer, turns)
        result = []
        batch_size = (
            self._detect_batch_size(tokenizer, prompts)
            if batch_size is None
            else batch_size
        )

        for prompt in prompts:
            print(f"Prompt: {prompt}")

        self.model.eval()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            with torch.no_grad():
                from tqdm import tqdm

                for i in tqdm(range(0, len(prompts), batch_size)):
                    batch_prompts = prompts[i : i + batch_size]
                    inputs = self._tokenize(
                        tokenizer,
                        batch_prompts,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model.generate(
                        **inputs,
                        temperature=temperate,
                        do_sample=True,
                        max_new_tokens=self.max_new_tokens,
                    )
                    result.extend(
                        [
                            tokenizer.decode(output, skip_special_tokens=True)
                            for output in outputs
                        ]
                    )

            return result


def predict_samples(
    conversations: list[dict],
    model: str,
    temp: float,
    max_tokens: int,
    chat_template: str | None,
    batch_size: int | None,
):
    with stub.run():
        x = Model(
            model,
            chat_template=chat_template,
            device="cuda",
            max_new_tokens=max_tokens,
        )
        return x.predict_basic.remote(conversations, temp, batch_size)
