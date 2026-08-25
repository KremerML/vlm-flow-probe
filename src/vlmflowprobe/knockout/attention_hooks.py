"""LLaVA attention hooking utilities.

Provides functions to block or modify attention patterns in LLaVA models
by wrapping the self-attention forward method at specified layers.

Extracted from root-level methods.py to consolidate hook logic.
"""

import functools

import torch


def set_block_attn_add_hooks_llava(model, values_per_layer, coef_value=0, only_knockout_question=None, question_range=[]):
    def change_values(values, coef_val, only_knockout_question, question_range):
        def hook(module, input, output):
            if only_knockout_question:
                assert output.shape[-1] == len(values)
                output[:, question_range, :] = coef_val
            else:
                output[:, :, values] = coef_val
        return hook

    hooks = []
    for layer, values in values_per_layer.items():
        hooks.append(model.model.layers[layer].self_attn.o_proj.register_forward_hook(change_values(values, coef_value, only_knockout_question, question_range)))
    return hooks


def set_block_attn_hooks_llava(model, from_to_index_per_layer, opposite=False, block_desc=None):
    def wrap_attn_forward(forward_fn, model_, from_to_index_, opposite_, block_desc_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):

            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for (k, v) in kwargs.items():
                new_kwargs[k] = v

            num_tokens = kwargs["position_ids"][0][-1].item()+1
            q_length = kwargs["hidden_states"][0].size(0)

            if q_length==1:
                if block_desc_.split("->")[-1]=="Last":
                    from_to_index=[(0, t) for _, t in from_to_index_]
                else:
                    from_to_index = []

            else:
                from_to_index=from_to_index_

            if opposite_:
                if q_length == 1:
                    attn_mask = torch.zeros((q_length, num_tokens), dtype=torch.uint8)
                else:
                    attn_mask = torch.tril(torch.zeros((q_length, num_tokens), dtype=torch.uint8))

                if from_to_index !=[]:
                    rows, cols = zip(*from_to_index)
                    attn_mask[rows, cols] = 1
            else:
                if q_length == 1:
                    attn_mask = torch.ones((q_length, num_tokens), dtype=torch.uint8)
                else:
                    attn_mask = torch.tril(torch.ones((q_length, num_tokens), dtype=torch.uint8))

                if from_to_index !=[]:
                    rows, cols = zip(*from_to_index)
                    attn_mask[rows, cols] = 0


            attn_mask = attn_mask.repeat(1, 1, 1, 1)

            attn_mask = attn_mask.to(dtype=model_.dtype)  # fp16 compatibility
            attn_mask = (1.0 - attn_mask) * torch.finfo(model_.dtype).min
            attn_mask = attn_mask.to(model_.device)
            new_kwargs["attention_mask"] = attn_mask
            return forward_fn(*new_args, **new_kwargs)

        return wrapper_fn

    hooks = []
    for i in from_to_index_per_layer.keys():
        hook = model.model.layers[i].self_attn.forward
        model.model.layers[i].self_attn.forward = wrap_attn_forward(model.model.layers[i].self_attn.forward,
                                                                model, from_to_index_per_layer[i], opposite, block_desc)
        hooks.append((i, hook))

    return hooks


def remove_wrapper_llava(model, hooks):
    for i, hook in hooks:
        model.model.layers[i].self_attn.forward = hook


def set_block_mlp_hooks_llava(model, values_per_layer, coef_value=0, only_knockout_question=None, question_range=[]):
    def change_values(values, coef_val, only_knockout_question, question_range):
        def hook(module, input, output):
            if only_knockout_question:
                assert output.shape[-1] == len(values)
                output[:, question_range, :] = coef_val
            else:
                output[:, :, values] = coef_val
        return hook

    hooks = []
    for layer, values in values_per_layer.items():
        hooks.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(change_values(values, coef_value, only_knockout_question, question_range)))
    return hooks


def trace_with_attn_block_llava(
        model,
        inp,
        from_to_index_per_layer,
        first_answer_token_id,
        block_desc,
        model_name
):
    with torch.inference_mode():
        block_attn_hooks = set_block_attn_hooks_llava(model, from_to_index_per_layer, block_desc=block_desc)
        output_details = model.generate(**inp)
        answer_token_id = output_details['sequences']
        logits_first_answer_token = output_details['scores'][0]
        remove_wrapper_llava(model, block_attn_hooks)

    [base_score_first] = torch.softmax(logits_first_answer_token, dim=-1)[0][first_answer_token_id]

    return base_score_first
