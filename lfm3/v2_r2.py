"""Run the archived two-pass paired-text decoder for the LFM3 v2 adapter."""

import torch


@torch.inference_mode()
def rewrite_v2_r2(tokenizer, model, source):
    text = source.removeprefix("\ufeff").strip()
    if not 100 <= len(text.split()) <= 3500:
        raise ValueError("The historical benchmark accepts 100 to 3500 whitespace-delimited words")

    for seed in (42, 43):
        torch.manual_seed(seed)
        prompt = f"<source_text>\n{text}\n</source_text>\n<target_text>\n"
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]
        limit = min(16000, max(512, 2 * prompt_length + 128))
        output = model.generate(
            **inputs,
            max_new_tokens=limit,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        tokens = output[0, prompt_length:]
        if len(tokens) >= limit and tokens[-1].item() != tokenizer.eos_token_id:
            raise ValueError("Generation reached the token limit before the end of the edit")
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip().removesuffix("</target_text>").strip()
        if not text:
            raise ValueError("Empty LFM generation")
    return text
