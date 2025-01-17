import queue
import threading
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from torch.utils.data import Dataset, DataLoader


def filter_metadata(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if isinstance(data[key], (dict, list)):
                new_data[key] = filter_metadata(value)
            elif isinstance(data[key], (int, float, bool, str)):
                new_data[key] = value
        return new_data
    elif isinstance(data, list):
        new_data = []
        for item in data:
            if isinstance(item, (dict, list)):
                new_data.append(filter_metadata(item))
            elif isinstance(item, (int, float, bool, str)):
                new_data.append(item)
        return new_data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


class BaseEvalDataset(Dataset, metaclass=ABCMeta):

    BENCHMARK_TYPE: str = None

    def __init__(
        self,
        data_root: str,
        processor: Callable,
        num_splits: int = 1,
        split_idx: int = 0,
        fps: int = 1,
        max_frames: int = 180,
    ) -> None:
        assert split_idx < num_splits, f"split_idx ({split_idx}) should be less than num_splits ({num_splits})"
        self.processor = processor
        self.fps = fps
        self.max_frames = max_frames

        self.data_dict = self.load_data(data_root)

        aggregated_data = dict()
        for data_id, meta_data in self.data_dict.items():
            video_path = meta_data["video_path"]
            start_time = meta_data["start_time"]
            end_time = meta_data["end_time"]
            aggregated_data_id = f"{video_path}_{start_time}_{end_time}"
            if aggregated_data_id not in aggregated_data:
                aggregated_data[aggregated_data_id] = {
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "data_ids": [data_id],
                }
            else:
                aggregated_data[aggregated_data_id]["data_ids"].append(data_id)

        aggregated_data_list = [x for _, x in aggregated_data.items()]
        self._aggregated_data_list = aggregated_data_list[split_idx::num_splits]

    @property
    def n_samples(self) -> int:
        return sum([len(x["data_ids"]) for x in self._aggregated_data_list])

    def __len__(self) -> int:
        return len(self._aggregated_data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        aggregated_data = self._aggregated_data_list[idx]

        # TODO: support fps and max_frames
        video = self.processor(
            aggregated_data["video_path"],
            s=aggregated_data["start_time"],
            e=aggregated_data["end_time"],
        )

        instructions = [self.generate_instruction(data_id, video) for data_id in aggregated_data["data_ids"]]
        data = {
            "video": video,
            "data_ids": aggregated_data["data_ids"],
            "instructions": instructions,
        }

        return data

    @abstractmethod
    def load_data(self, data_root) -> Dict[Union[int, str], Any]:
        """
        Load the dataset meta data.

        Args:
            data_root (str): path to the dataset.

        Returns:
            data_dict (Dict[Union[int, str], Any]): dataset meta data, with data_id as key.
            example:
            {
                0: {
                    # required fields for data loading
                    "video_path": os.path.join(video_folder, data["video"]),
                    "start_time": data["start"] if task_info[3] else None,
                    "end_time": data["end"] if task_info[3] else None,
                    # required fields for evaluation
                    "task_type": task_name,
                    "ground_truth": answer_idx,
                    # custom fields for instruction generation and post processing
                    "question": data["question"],
                    "options": options,
                    "option_letters": option_letters,
                }
                ...
            }
        """
        pass

    @abstractmethod
    def generate_instruction(self, data_id: Union[int, str], video: Any) -> Union[str, Dict[str, str]]:
        """
        Generate instruction(s) for model inference.

        Args:
            data_id (Union[int, str]): identifier of the data.

        Returns:
            instruction (Union[str, Dict[str, str]]): instruction(s) for model inference.
        """
        pass

    @abstractmethod
    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        """
        Process the original model responses to desired format for evaluation and visualization.

        Args:
            data_id (Union[int, str]): identifier of the data.
            response (str): model response.

        Returns:
            result (Any): processed model response for evaluation.
        """
        pass

    def evaluate(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics according to predictions and ground-truths.

        Args:
            results (List[Dict[str, Any]]): list of processed model responses.

        Returns:
            metrics (Dict[str, float]): evaluation metrics.
            infos (List[Dict[str, Any]]): evaluation information for visualization.
        """
        assert self.BENCHMARK_TYPE is not None, "BENCHMARK_TYPE is not defined."
        if self.BENCHMARK_TYPE == "mcqa":
            return self._eval_mcqa(results)
        else:
            raise NotImplementedError(f"Unsupported benchmark type: {self.BENCHMARK_TYPE}")

    def _eval_mcqa(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics for multiple-choice question answering tasks.

        Args:
            results (List[Dict[str, Any]]): list of processed model responses.

        Returns:
            metrics (Dict[str, float]): evaluation metrics.
            infos (List[Dict[str, Any]]): evaluation information for visualization.
        """
        samples = defaultdict(list)
        infos = []

        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            ground_truth = meta_data["ground_truth"]
            task_type = meta_data["task_type"]
            matching = data["prediction"] == meta_data["ground_truth"]
            
            samples[task_type].append(int(matching))
            infos.append(
                {
                    **data,
                    "ground_truth": ground_truth,
                    "matching": matching,
                    "task_type": task_type,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        task_types = sorted(samples.keys())
        metrics = {x: sum(samples[x]) / len(samples[x]) * 100 for x in task_types}

        overall_samples = sum(samples.values(), [])
        overall_acc = sum(overall_samples) / len(overall_samples) * 100
        metrics["Overall"] = overall_acc

        infos = [metrics] + infos
        return metrics, infos


class BackgroundGenerator(threading.Thread):
    """
    the usage is below
    >> for batch in BackgroundGenerator(my_minibatch_iterator):
    >>    doit()
    More details are written in the BackgroundGenerator doc
    >> help(BackgroundGenerator)
    """

    def __init__(self, generator, local_rank, max_prefetch=10):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may raise GIL and zero-out the
        benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it
        outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving
        URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep
        stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until
        one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work
        slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size
        unless dequeued quickly enough.
        """
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class CUDADataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.stream = torch.cuda.Stream(local_rank) # create a new cuda stream in each process
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            # avoid re-entrance or ill-conditioned thread state
            return

        # Set exit event to True for background threading stopping
        self.iter.exit_event.set()

        # Exhaust all remaining elements, so that the queue becomes empty,
        # and the thread should quit
        for _ in self.iter:
            pass

        # Waiting for background thread to quit
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            frames = self.batch['video'][0][0][0]
            for idx, frame in enumerate(frames):
                frames[idx]['pixel_values'] = frame['pixel_values'].to(device=self.local_rank, non_blocking=True)
                frames[idx]['image_grid_thw'] = frame['image_grid_thw'].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)  # wait tensor to put on GPU
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        # If the dataloader is to be freed, shutdown its BackgroundGenerator
        self._shutdown_background_thread()
