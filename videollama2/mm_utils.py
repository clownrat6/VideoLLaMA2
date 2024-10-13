import ast
import math
import base64
from io import BytesIO

import torch
import imageio
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from transformers import StoppingCriteria

from .constants import NUM_FRAMES, MAX_FRAMES, NUM_FRAMES_PER_SECOND, MMODAL_INDEX_TOKEN, IMAGE_TOKEN_INDEX


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def frame_expansion(frame_list, n):
    assert len(frame_list) == n * n
    width, height = frame_list[0].width, frame_list[0].height
    expanded_width = n * width
    expanded_height = n * height
    expanded_frame = Image.new('RGB', (expanded_width, expanded_height))
    for i in range(n):
        for j in range(n):
            frame = frame_list[i * n + j]
            coordinate = (j*width, i*height)
            expanded_frame.paste(frame, coordinate)
    return expanded_frame


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    #print("Current image_aspect_ratio:", image_aspect_ratio)
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def process_videos(frames, image_processor, model_cfg):
    # this function only used during inference
    # image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    # new_frames = []
    # print("Current image_aspect_ratio:", image_aspect_ratio)
    # if image_aspect_ratio == 'pad':
    #     for image in frames:
    #         image = Image.fromarray(image)
    #         image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    #         image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    #         new_frames.append(image)
    # else:
    #     return image_processor(frames, return_tensors='pt')['pixel_values']
    # if all(x.shape == new_frames[0].shape for x in new_frames):
    #     new_frames = torch.stack(new_frames, dim=0)
    new_frames = image_processor.preprocess(frames, return_tensors='pt')['pixel_values']  # do not pad for video frames
    return new_frames


def create_photo_grid(arr, rows=None, cols=None):
    """
    Create a photo grid from a 4D numpy array with shape [t, h, w, c].

    Parameters:
        arr (numpy.ndarray): Input array with shape [t, h, w, c].
        rows (int): Optional. Number of rows in the grid. If not set, it will be determined based on `cols` or the square root of `t`.
        cols (int): Optional. Number of columns in the grid. If not set, it will be determined based on `rows` or the square root of `t`.

    Returns:
        numpy.ndarray: A 3D numpy array representing the photo grid.
    """

    if isinstance(arr, list):
        if isinstance(arr[0], Image.Image):
            arr = np.stack([np.array(img) for img in arr])
        elif isinstance(arr[0], np.ndarray):
            arr = np.stack(arr)
        else:
            raise ValueError("Invalid input type. Expected list of Images or numpy arrays.")

    t, h, w, c = arr.shape
    
    # Calculate the number of rows and columns if not provided
    if rows is None and cols is None:
        rows = math.ceil(math.sqrt(t))
        cols = math.ceil(t / rows)
    elif rows is None:
        rows = math.ceil(t / cols)
    elif cols is None:
        cols = math.ceil(t / rows)

    # Check if the grid can hold all the images
    if rows * cols < t:
        raise ValueError(f"Not enough grid cells ({rows}x{cols}) to hold all images ({t}).")
    
    # Create the grid array with appropriate height and width
    grid_height = h * rows
    grid_width = w * cols
    grid = np.zeros((grid_height, grid_width, c), dtype=arr.dtype)
    
    # Fill the grid with images
    for i in range(t):
        row_idx = i // cols
        col_idx = i % cols
        grid[row_idx*h:(row_idx+1)*h, col_idx*w:(col_idx+1)*w, :] = arr[i]
    
    return grid


def process_image(image_path, processor, aspect_ratio='pad', num_frames=NUM_FRAMES, image_grid=False):
    image = Image.open(image_path).convert('RGB')

    if image_grid:
        pg = np.stack([np.array(image)] * num_frames)
        grid_h = grid_w = math.ceil(math.sqrt(num_frames))
        pg = create_photo_grid(pg, grid_h, grid_w)
        images = [pg, np.array(image)]
    else:
        images = [np.array(image)]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return images


def process_video(video_path, processor, aspect_ratio='pad', num_frames=NUM_FRAMES, image_grid=False, sample_scheme='uniform'):
    def frame_sample(duration, mode='uniform', local_fps=None):
        if mode == 'uniform':
            # Calculate the size of each segment from which a frame will be extracted
            seg_size = float(duration - 1) / num_frames

            frame_ids = []
            for i in range(num_frames):
                # Calculate the start and end indices of each segment
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                # Append the middle index of the segment to the list
                frame_ids.append((start + end) // 2)

            return frame_ids
            # NOTE: old version
            # return np.linspace(0, duration-1, num_frames, dtype=int)
        elif mode == 'fps':
            assert local_fps is not None
            segment_len = min(local_fps // NUM_FRAMES_PER_SECOND, duration)
            return np.arange(segment_len // 2, duration, segment_len, dtype=int)
        else:
            raise ImportError(f'Unsupported frame sampling mode: {mode}')

    if isinstance(video_path, str):
        if video_path.endswith('.gif'):
            video_gif = imageio.get_reader(video_path)
            duration, local_fps = len(video_gif), 10

            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)
            video_data = [frame for index, frame in enumerate(video_gif) if index in frame_id_list]
        # added by lixin4ever, include the support of .webm files from sthsthv2
        elif video_path.endswith('.webm'):
            video_webm = VideoFileClip(video_path)
            video_frames = np.array(list(video_webm.iter_frames()))

            duration, local_fps = len(video_frames), video_webm.fps

            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)
            video_data = video_frames[frame_id_list]
        else:
            # NOTE: num_threads=1 is required to avoid deadlock in multiprocessing
            decord_vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) 
            duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
        
            frame_id_list = frame_sample(duration, mode=sample_scheme, local_fps=local_fps)
            # limit the max input frames
            if len(frame_id_list) > MAX_FRAMES:
                frame_id_list = np.linspace(0, duration-1, MAX_FRAMES, dtype=int)
            try:
                video_data = decord_vr.get_batch(frame_id_list).numpy()
            except:
                video_data = decord_vr.get_batch(frame_id_list).asnumpy()

            # if self.data_args.use_temp_aug:
            #     frame_id_list = np.linspace(0, duration-1, num_frames * 2 * 2, dtype=int)
            #     video_data = decord_vr.get_batch(frame_id_list)
            #     video_frames = [Image.fromarray(f) for f in video_data.numpy()]
            #     chunked_video_frames = chunk_list(video_frames, 2*2)
            #     video_data = [frame_expansion(frame_list, 2) for frame_list in chunked_video_frames]
    elif isinstance(video_path, np.ndarray):
        assert len(video_path) == num_frames
        video_data = video_path
    elif isinstance(video_path, list):
        assert len(video_path) == num_frames
        video_data = np.stack([np.array(x) for x in video_path])

    if image_grid:
        grid_h = grid_w = math.ceil(math.sqrt(num_frames))
        pg = create_photo_grid(video_data, grid_h, grid_w)
        video_data = [pg, *video_data]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    else:
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']

    return video


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(f'<{MMODAL_INDEX_TOKEN[MMODAL_token_index].lower()}>')]
    num_prompt_chunks = len(prompt.split(f'<{MMODAL_INDEX_TOKEN[MMODAL_token_index].lower()}>'))

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [MMODAL_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
