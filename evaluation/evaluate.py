import argparse
import os
import os.path as osp
import queue
import threading
import traceback
from typing import Any, Dict, List, Union
from collections.abc import MutableMapping, Sequence

import json
import torch
import torch.distributed as dist
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import sys
sys.path.append(".")
from evaluation.register import INFERENCES
from evaluation.datasets import build_dataset
from videollama2.utils import disable_torch_init
from videollama2 import model_init, mm_infer


def to_cuda(packed_data):
    if isinstance(packed_data, torch.Tensor):
        packed_data = packed_data.to(device="cuda", non_blocking=True)
    elif isinstance(packed_data, (int, float, str, bool, complex)):
        packed_data = packed_data
    elif isinstance(packed_data, MutableMapping):
        for key, value in packed_data.items():
            packed_data[key] = to_cuda(value)
    elif isinstance(packed_data, Sequence):
        for i, value in enumerate(packed_data):
            packed_data[i] = to_cuda(value)
    return packed_data


class CUDADataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = torch.cuda.Stream() # create a new cuda stream in each process
        self.queue = queue.Queue(64)

    def preload(self):
        batch = next(self.iter)
        if batch is None:
            return None
        torch.cuda.current_stream().wait_stream(self.stream)  # wait tensor to put on GPU
        with torch.cuda.stream(self.stream):
            batch = to_cuda(batch)
        self.queue.put(batch)

    def __iter__(self):
        # setting a queue for storing prefetched data
        self.queue.queue.clear()
        # reset data iterator
        self.iter = super().__iter__()
        # starting a new thread to prefetch data
        def data_to_cuda_then_queue():
            while True:
                try:
                    self.preload()
                except StopIteration:
                    break
            # NOTE: end flag for the queue
            self.queue.put(None)

        self.thread = threading.Thread(target=data_to_cuda_then_queue, args=())
        self.thread.daemon = True

        (self.preload() for _ in range(16))
        self.thread.start()
        return self

    def __next__(self):
        next_item = self.queue.get()
        # NOTE: __iter__ will be stopped when __next__ raises StopIteration 
        if next_item is None:
            raise StopIteration
        return next_item

    def __del__(self):
        # NOTE: clean up the thread
        try:
            self.thread.join(timeout=10)
        finally:
            if self.thread.is_alive():
                self.thread.terminate()
        # NOTE: clean up the stream
        self.stream.synchronize()
        # NOTE: clean up the queue
        self.queue.queue.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)

    parser.add_argument("--data-root", "--data_root", type=str, required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", "--max_frames", type=int, default=180)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)

    parser.add_argument("--save-path", "--save_path", type=str, default=None)

    return parser.parse_args()


def show_metrics(metrics: Dict[str, Any], benchmark: str):
    if all(isinstance(metric, dict) for metric in metrics.values()):
        for task_name, metric in metrics.items():
            show_metrics(metric, f"{benchmark}_{task_name}")
    elif all(isinstance(metric, (int, float)) for metric in metrics.values()):
        table = PrettyTable(["Task Type", "Accuracy"])
        for task_name, metric in metrics.items():
            table.add_row([task_name, round(metric, 2)])
        table.align["Task Type"] = "l"
        print(f"Results on {benchmark}:")
        print(table)
        print("\n")
    else:
        raise ValueError


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    disable_torch_init()
    model_init, mm_infer = INFERENCES(args.model_path)
    model, processor, tokenizer = model_init(args.model_path, device_map={"": f"cuda:{local_rank}"})

    dataset = build_dataset(
        args.benchmark,
        data_root=args.data_root,
        processor=processor["video"],
        num_splits=dist.get_world_size(),
        split_idx=dist.get_rank(),
        fps=args.fps,
        max_frames=args.max_frames,
    )
    dataloader = CUDADataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=lambda x: x[0],  # asume the batch_size is always 1
        pin_memory=True,
    )

    results = []
    for data in tqdm(dataloader, desc=f"Rank {local_rank}", total=len(dataloader), position=local_rank):
        data_ids = data["data_ids"]
        instructions = data["instructions"]
        for data_id, instruction in zip(data_ids, instructions):
            try:
                response = mm_infer(
                    data["video"],
                    instruction,
                    model=model,
                    tokenizer=tokenizer,
                    modal="video",
                    do_sample=False,
                )
                prediction = dataset.process_response(data_id, response)
            except Exception as e:
                traceback.print_exc()
                print(f"Error in data_id: {data_id}")
                exit(0)

            results.append(
                {
                    "data_id": data_id,
                    "response": response,
                    "prediction": prediction,
                }
            )

    assert len(results) == dataset.n_samples

    del model, data
    torch.cuda.empty_cache()
    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.gather_object(
        obj=results,
        object_gather_list=gathered_results if dist.get_rank() == 0 else None,
        dst=0,
    )

    if dist.get_rank() == 0:
        results = sum(gathered_results, [])
        metrics, infos = dataset.evaluate(results)

        print("\n" * dist.get_world_size())  # prevent unexpected progress bars
        show_metrics(metrics, args.benchmark)

        if args.save_path:
            os.makedirs(osp.dirname(args.save_path), exist_ok=True)
            if args.save_path.endswith(".json"):
                with open(args.save_path, "w") as f:
                    json.dump(infos, f, indent=4)
            elif args.save_path.endswith(".jsonl"):
                with open(args.save_path, "w") as f:
                    for info in infos:
                        f.write(json.dumps(info) + "\n")
            else:
                raise ValueError("Unsupported file format.")


if __name__ == "__main__":
    main()
