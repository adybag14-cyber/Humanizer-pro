"""Load the editing checkpoint and format inputs consistently for training and inference."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "LiquidAI/LFM3-1.7B"
BASE_REVISION = "0dc4268b561499c648c7ba7c9466bf3d8d01517a"
SYSTEM = (
    "Edit the text for clarity. Preserve every fact, number, condition, uncertainty and responsible "
    "party. Do not add facts. Return only the revised passage."
)


def load(model_id, revision=None):
    options = {"trust_remote_code": True, "revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0", **options
    )
    return tokenizer, model


def prompt_ids(tokenizer, source):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": source}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer.encode(prompt, add_special_tokens=False)


@torch.inference_mode()
def rewrite(tokenizer, model, source):
    tokens = torch.tensor([prompt_ids(tokenizer, source)], device=model.device)
    result = model.generate(
        input_ids=tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
        max_new_tokens=384, pad_token_id=tokenizer.eos_token_id,
    )
    reply = result[0, tokens.shape[1]:]
    if len(reply) == 384 and reply[-1].item() != tokenizer.eos_token_id:
        raise ValueError("Generation reached the token limit before the end of the edit")
    return tokenizer.decode(reply, skip_special_tokens=True).strip()
