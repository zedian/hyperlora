import torch
from typing import Optional
from transformers import Trainer, data, Seq2SeqTrainer
from torch.utils.data.dataloader import DataLoader
from src.sampling_utils import MultiDialectBatchSampler

class RobertaTrainer(Trainer):
    def __init__(self, data_collator, dataset_sizes=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_sizes = dataset_sizes
        self.data_collator = data_collator
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        num_replicas = 1
        rank = 0
        self.args.temperature = 5
        return MultiDialectBatchSampler(self.dataset_sizes, self.args.train_batch_size,
                                     self.args.temperature, rank=rank,
                                     num_replicas=num_replicas)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        multitask_sampler = self._get_train_sampler()
        return DataLoader(self.train_dataset, batch_sampler=multitask_sampler,
                          collate_fn=self.data_collator)

class T5Trainer(Seq2SeqTrainer):
    def __init__(self, data_collator, dataset_sizes=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_sizes = dataset_sizes
        self.data_collator = data_collator
        self.args.generation_max_length = 128
        self.args.generation_num_beams = 5
        self.args.predict_with_generate = True
        self._gen_kwargs = kwargs
        self._gen_kwargs["max_length"] = self.args.generation_max_length
        self._gen_kwargs["num_beams"] = self.args.generation_num_beams = 5
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        num_replicas = 1
        rank = 0
        self.args.temperature = 5
        return MultiDialectBatchSampler(self.dataset_sizes, self.args.train_batch_size,
                                     self.args.temperature, rank=rank,
                                     num_replicas=num_replicas)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        multitask_sampler = self._get_train_sampler()
        return DataLoader(self.train_dataset, batch_sampler=multitask_sampler,
                          collate_fn=self.data_collator)