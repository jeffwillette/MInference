import gc
import math
import os
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Subset
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


TRITON_DEBUG = os.getenv("TRITON_DEBUG", "0") == "1"
WANDB_DISABLED = os.getenv("WANDB_MODE", "none") == "disabled"


def evaluate(model, tokenizer):

    setattr(model, "no_lm_head", True)
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    layer = model.model.layers[len(model.model.layers) // 2]
    layer.mlp = nn.Identity()
    layer.input_layernorm = nn.Identity()
    layer.post_attention_layernorm = nn.Identity()
    layer.self_attn.q_proj = nn.Identity()
    layer.self_attn.k_proj = nn.Identity()
    layer.self_attn.v_proj = nn.Identity()
    layer.self_attn.o_proj = nn.Identity()
    layer = layer.cuda()

    metrics = {"hip": [], "window": [], "fa2": []}
    with torch.no_grad():
        # TODO:
        # - make inputs with tokenizer
        # - swap in attention types
        # - run warmup foir kernel
        # - benchmark a forward pass on one attention layer
        # - evaluate:
        #  - hip: w 64, 128, 256, 512, 1024, 2048
        #  - window: w 64, 128, 256, 512, 1024, 2048
        #  - fa2

        # env vars attention method: [postfixes to run]
        env_vars = {
            "flash_attention_2": ["recompute_dense-window_0-diff_1-w_64"],
        }

        for ctx_len in [2**i for i in range(15, 21)]:

            x = torch.randn(1, ctx_len, 4096,
                            dtype=torch.bfloat16, device="cuda:0")
            position_embeddings = (
                torch.randn(1, ctx_len, 128, device=x.device, dtype=x.dtype),
                torch.randn(1, ctx_len, 128, device=x.device, dtype=x.dtype),
            )
            position_ids = torch.arange(ctx_len, device=x.device).unsqueeze(0)

            torch.cuda.synchronize()

            for attn in env_vars.keys():
                for postfix in env_vars[attn]:
                    os.environ["ATTENTION_IMPLEMENTATION"] = attn
                    os.environ["USE_ATTN_POSTFIX"] = postfix

                    prev_attn = model.config._attn_implementation
                    if "hip" in attn:
                        model.config._attn_implementation = "recompute_dense"
                    elif "flash_attention_2" in attn:
                        model.config._attn_implementation = "flash_attention_2"
                    elif "minference" in attn:
                        print("minference")
                    else:
                        raise Exception()

                    times = []
                    for i in range(10):
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        # Record time
                        start_event.record()
                        layer(
                            x,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )
                        end_event.record()

                        # Wait for the events to be recorded
                        torch.cuda.synchronize()

                        # Compute elapsed time in milliseconds
                        if i >= 3:
                            times.append(start_event.elapsed_time(end_event))

                    model.config._attn_implementation = prev_attn

                    if len(times) > 0:
                        print(
                            f"{ctx_len=} {attn=} {postfix=}: elapsed time: {sum(times) / len(times):.3f} ms"
                        )


def init_model():
    device = "cuda:0"

    ALL_ATTENTION_FUNCTIONS.update({"hip_attention": (lambda x: x)})
    ALL_ATTENTION_FUNCTIONS.update({"sdpa_rectangle": (lambda x: x)})
    ALL_ATTENTION_FUNCTIONS.update({"recompute": (lambda x: x)})

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    attn_implementation = os.environ.get(
        "ATTN_IMPLEMENTATION", "hip_attention")

    if "minference" in attn_implementation:
        from minference import MInference

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

        minference_patch = MInference(attn_implementation, model_name)
        model = minference_patch(model)
    else:
        raise ValueError()

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = init_model()
    tokenizer.pad_token = tokenizer.eos_token

    model = model.cpu()
    metrics = evaluate(model, tokenizer)
