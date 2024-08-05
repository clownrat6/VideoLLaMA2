import os
import copy
import warnings
import shutil
from functools import partial

import torch

from .model import load_pretrained_model
from .mm_utils import process_image, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP


def model_init(model_path=None):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)

    if tokenizer.unk_token is not None: 
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    return model, partial(process_video, aspect_ratio=None, processor=processor, num_frames=num_frames), tokenizer


def infer(model, video, instruct, tokenizer, do_sample=False):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        version (str): conversation template version.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    modal_token = DEFAULT_VIDEO_TOKEN
    instruct = modal_token + '\n' + instruct

    # 2. vision preprocess (load & transform image or video).
    tensor = video.half().cuda()

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + [{'role': 'user', 'content': instruct}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    tensor = [(tensor, modal_token)]

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def x_infer(video, question, model, tokenizer, mode='vanilla', do_sample=False):
    if mode == 'mcqa':
        instruction = f'{question}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample)
    elif mode == 'openend':
        instruction = f'{question}\nAnswer the question using a single word or a short phrase with multiple words.'
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample)
    elif mode == 'vanilla':
        instruction = question
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample)
