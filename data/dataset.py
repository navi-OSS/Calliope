"""
Dataset loading for curriculum training (TinyStories -> bAbI -> SimpleStories).
"""
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Optional


class LMDataset(Dataset):
    """
    Generic Language Modeling dataset.
    Handles different data sources and formats them for autoregressive training.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "roneneldan/TinyStories",
        dataset_config: Optional[str] = None,
        max_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None,
        streaming: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.streaming = streaming
        
        # Load dataset
        print(f"📚 Loading {dataset_name} ({dataset_config if dataset_config else 'default'}) [Streaming={streaming}]...")
        self.data = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
        
        if not streaming and max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(self.data))))
        elif streaming and max_samples is not None:
            self.data = self.data.take(max_samples)
    
    def __len__(self) -> int:
        if self.streaming:
            raise NotImplementedError("Streaming dataset has no length. Use IterableDataset logic.")
        return len(self.data)
    
    def _format_babi(self, item: dict) -> str:
        """Format bAbI task: Story + Question + Answer."""
        try:
            # Handle list-based story (facebook/babi_qa)
            if isinstance(item.get("story"), dict) and "text" in item["story"]:
                story_text = " ".join(item["story"]["text"])
            # Handle string-based story (Muennighoff/babi or others)
            elif isinstance(item.get("story"), str):
                story_text = item["story"]
            elif isinstance(item.get("context"), str):
                story_text = item["context"]
            else:
                story_text = ""
                
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not question and not answer:
                return str(item.get("text", item))
                
            return f"{story_text} {question} {answer}".strip()
        except Exception:
            return str(item)

    def process_item(self, item: dict) -> dict:
        """Process a single item (format + tokenize)."""
        # specialized formatting based on dataset name
        name_lower = self.dataset_name.lower()
        if "babi" in name_lower:
            text = self._format_babi(item)
        elif "simplestories" in name_lower:
            text = item.get("text", item.get("story", item.get("content", "")))
        elif "writingprompts" in name_lower:
            prompt = item.get("prompt", "")
            story = item.get("story", "")
            text = f"Prompt: {prompt}\nStory: {story}"
        else:
            text = item.get("text", "")
            
        if not text:
            for val in item.values():
                if isinstance(val, str) and len(val) > 10:
                    text = val
                    break
            
        encoding = self.tokenizer(
            str(text),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }

    def __getitem__(self, idx: int) -> dict:
        if self.streaming:
            raise NotImplementedError("Streaming dataset does not support index access.")
        return self.process_item(self.data[idx])

    def __iter__(self):
        if not self.streaming:
            raise NotImplementedError("Map-style dataset should use __getitem__.")
        for item in self.data:
            yield self.process_item(item)


class MixtureDataset(IterableDataset):
    """
    Dataset that mixes multiple LMDatasets (Stream or Map) with specified weights.
    Always functions as an IterableDataset for consistency.
    """
    
    def __init__(
        self,
        datasets: list[Dataset],
        weights: list[float],
        samples_per_epoch: int = 10000,
    ):
        assert len(datasets) == len(weights)
        self.datasets = datasets
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.samples_per_epoch = samples_per_epoch
        
    def __iter__(self):
        # Create iterators for all datasets
        # If map-style, we create a random sampler iterator
        iterators = []
        for ds in self.datasets:
            if hasattr(ds, 'streaming') and ds.streaming:
                iterators.append(iter(ds))
            else:
                # For map-style, create an infinite random sampler
                iterators.append(self._infinite_map_iterator(ds))
        
        count = 0
        while count < self.samples_per_epoch:
            # Select dataset
            dataset_idx = torch.multinomial(self.weights, 1).item()
            try:
                yield next(iterators[dataset_idx])
                count += 1
            except StopIteration:
                # If a stream ends, we could restart it, but for now we continue
                # In robust training, we should cycle streams. 
                # Simplest fix: Just pick another dataset this time?
                # Or simplistic: strict StopIteration means end of training step.
                pass

    def _infinite_map_iterator(self, dataset):
        """Helper to treat map-style dataset as infinite stream."""
        while True:
            idx = torch.randint(0, len(dataset), (1,)).item()
            yield dataset[idx]

    def __len__(self):
        return self.samples_per_epoch


def get_dataloader(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "roneneldan/TinyStories",
    dataset_config: Optional[str] = None,
    batch_size: int = 8,
    max_length: int = 512,
    split: str = "train",
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    streaming: bool = False,
) -> DataLoader:
    """Create a DataLoader for the specified dataset."""
    dataset = LMDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_length=max_length,
        split=split,
        max_samples=max_samples,
        streaming=streaming,
    )
    
    # If using Mixture or Streaming, we cannot use shuffle=True or drop_last=True in standard ways sometimes
    # But for IterableDataset, shuffle is ignored by DataLoader (must be done in buffer).
    # MixtureDataset does stochastic sampling so it IS shuffled.
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
