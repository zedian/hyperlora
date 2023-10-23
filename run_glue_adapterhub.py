from collections import defaultdict
import logging
import sys
import numpy as np
import os
import random
import copy
import torch
import torch.nn as nn
import inspect
import json
import evaluate

from typing import Optional

import datasets
import transformers
# import transformers.adapters.composition as  ac
from transformers import (
    AutoConfig,
    AutoAdapterModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5AdapterModel,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.adapters import AdapterConfig, AdapterTrainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from peft import get_peft_config, get_peft_model, LoraConfig, PeftModel
from peft.tuners.lora import LoraModel

from datasets import concatenate_datasets, interleave_datasets, load_dataset, load_metric

from src.data_utils import DataTrainingArguments, Preprocessing, dialect_data_collator, label_data_collator
from src.model_utils import ModelArguments, RobertaLoraWrapper, IdentityWrapper
# from old_src.model_utils import IdentityWrapper
# from src.train_utils import get_params
from src.utils import task_to_keys

from src.trainer_utils import RobertaTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.3")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)

def calculate_statistics(vectors):
    num_vectors = len(vectors)
    vector_length = len(vectors[0])  # Assuming all vectors have the same length
    
    # Calculate the percentage of 1s in each dimension
    dimension_percentages = []
    for dim in range(vector_length):
        count_ones = sum(vector[dim] for vector in vectors)
        percentage_ones = (count_ones / num_vectors) * 100
        dimension_percentages.append(percentage_ones)
    
    # Calculate the percentage of 0 vectors in the list
    count_zero_vectors = sum(1 for vector in vectors if sum(vector) != 0)
    percentage_zero_vectors = (count_zero_vectors / num_vectors) * 100
    
    count_dimensions = sum(1 for d in dimension_percentages if d != 0)
    dimension_percentages = sorted(dimension_percentages)[-10:]
    return dimension_percentages, percentage_zero_vectors, count_dimensions

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    from evaluate import evaluator
    task_evaluator = evaluator("text-classification")
    metrics = task_evaluator.prepare_metric("accuracy")

    valid_str = "validation_matched" if data_args.task_name == "mnli" else "validation"
    test_str = "test_matched" if data_args.task_name == "mnli" else "test"

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        cache_name = "./cache/" + data_args.task_name
        if data_args.dialects != None:
            for dialect in data_args.dialects:
                cache_name = cache_name + dialect
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=cache_name,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        raw_datasets["train"] = raw_datasets["train"].select(range(1))
        raw_datasets[test_str] = raw_datasets[test_str].select(range(1))
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # if "t5" in model_args.model_name_or_path:
    #     ConfigClass = T5Config
    #     TokenizerClass = T5Tokenizer
    #     if model_args.task_adapter:
    #         ModelClass = T5AdapterModel
    #     else:
    #         ModelClass = T5ForConditionalGeneration
    # else:
    ConfigClass = AutoConfig
    TokenizerClass = AutoTokenizer
    if model_args.task_adapter:
        ModelClass = AutoAdapterModel
    else:
        ModelClass = AutoModelForSequenceClassification
    
    config = ConfigClass.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = TokenizerClass.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    model = ModelClass.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    if model_args.apply_adapter:
        adapter_config = AdapterConfig.load(model_args.adapter_name + "/adapter_config.json")
        model.load_adapter(model_args.adapter_name, config=adapter_config, load_as="tada_aave")

    if model_args.task_adapter:
        config_name = "pfeiffer"
        adapter_path = "WillHeld/pfadapter-roberta-base-"+data_args.task_name
        adapter_name = data_args.task_name or "glue"
        adapter_config = AdapterConfig.load(config_name)

        model.load_adapter(
                adapter_path,
                config=adapter_config,
                load_as=adapter_name,
            )
        # Freeze all model weights except of those of this adapter
        model.train_adapter([adapter_name])
        # Set the adapters to be used in every forward pass
        model.set_active_adapters([adapter_name])
        print(model.adapter_summary())

    if model_args.apply_lora and not training_args.do_train and training_args.do_eval:
        model = PeftModel.from_pretrained(model, "singe_lora_model", map_location=torch.device('cpu'))
        model.print_trainable_parameters()

    if model_args.apply_lora:
        peft_config = LoraConfig(
            task_type="SEQ_CLS", inference_mode=False, r=model_args.lora_rank, lora_alpha=model_args.lora_alpha, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if model_args.apply_hyperlora:
        model = RobertaLoraWrapper(model, data_args.dialects, model_args.lora_rank, model_args.hidden_adapter_dim, model_args.load_hypernet_weights)

    if not model_args.apply_hyperlora:
        model = IdentityWrapper(model)
    # With the wrapper, attributes should be passed to the inner model
    if hasattr(model, "model"):
        innermodel = model.model
    else:
        innermodel = model

    # print(model)
    percent_param = sum(p.numel() for p in model.parameters() if p.requires_grad)/ sum(p.numel() for p in model.parameters())
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"% Trainable parameters: {percent_param*100}")
    
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        innermodel.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in innermodel.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        innermodel.config.label2id = label_to_id
        innermodel.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        innermodel.config.label2id = {l: i for i, l in enumerate(label_list)}
        innermodel.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    prep = Preprocessing(sentence1_key, sentence2_key, tokenizer, padding, max_seq_length, label_to_id)

    all_datasets = {k: None for k in data_args.dialects}

    for i, dialect in enumerate(data_args.dialects):
        dialect_transform = prep.dialect_transform_factory(dialect)
        
        with training_args.main_process_first(desc="dataset map pre-processing"):
            if not data_args.load_dataset:
                all_datasets[dialect] = raw_datasets.map(
                    dialect_transform,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    num_proc=6,
                    desc="Transform Dataset Using Dialect Transformations",
                )
            else:
                all_datasets[dialect] = datasets.load_from_disk(data_args.save_dataset_path[i])

            if data_args.save_dataset:
                all_datasets[dialect].save_to_disk(data_args.save_dataset_path[i])

            if training_args.do_train:
                if "train" not in raw_datasets:
                    raise ValueError("--do_train requires a train dataset")

                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(raw_datasets["train"]), data_args.max_train_samples)
                    raw_datasets["train"] = raw_datasets["train"].select(range(max_train_samples))
                    all_datasets[dialect]["train"] = all_datasets[dialect]["train"].select(range(max_train_samples))

            # NOTE: Should we even eval on all dialects?
            # Possible solution: compute dialect specific accuracy
            if training_args.do_eval:
                if (
                    "validation" not in raw_datasets
                    and "validation_matched" not in raw_datasets
                ):
                    raise ValueError("--do_eval requires a validation dataset")

                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(raw_datasets[valid_str]), data_args.max_eval_samples)
                    raw_datasets[valid_str] = raw_datasets[valid_str].select(range(max_eval_samples))
                    all_datasets[dialect][valid_str] = all_datasets[dialect][valid_str].select(range(max_eval_samples))

            if (
                training_args.do_predict
                or data_args.task_name is not None
                or data_args.test_file is not None
            ):
                if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                    raise ValueError("--do_predict requires a test dataset")
                if data_args.max_predict_samples is not None:
                    max_predict_samples = min(
                        len(raw_datasets[test_str]), data_args.max_predict_samples
                    )
                    raw_datasets[test_str] = raw_datasets[test_str].select(range(max_predict_samples))
                    all_datasets[dialect][test_str] = all_datasets[dialect][test_str].select(range(max_predict_samples))

    first_dialect = data_args.dialects[0]
    if data_args.combine_sae:
        all_datasets["sae"] = raw_datasets
    else:
        raw_datasets["train"] = all_datasets[first_dialect]["train"].select([])
        raw_datasets[valid_str] = all_datasets[first_dialect][valid_str].select([])
        raw_datasets[test_str] = all_datasets[first_dialect][test_str].select([])
        all_datasets["sae"] = raw_datasets
    combine_on = "sae"

    all_datasets["combined"] = {"train":[], valid_str:[], "test":[]}
    for split in all_datasets["sae"]:
        dialect_list = []

        if data_args.combine_sae:
            dialect_list.append(all_datasets["sae"][split])
        for i, dialect in enumerate(data_args.dialects):
            dialect_list.append(all_datasets[dialect][split])
        if model_args.eval_dialect is not None and split==valid_str:
            all_datasets[combine_on][split] = all_datasets[model_args.eval_dialect][split]
        else:
            all_datasets[combine_on][split] = concatenate_datasets(dialect_list)
        
    dialect_datasets = all_datasets[combine_on]
    dataset_sizes = [len(all_datasets[first_dialect]["train"]) for _ in [*data_args.dialects]]

    raw_datasets = dialect_datasets.map(
        prep.preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets[valid_str]
    predict_dataset = raw_datasets[test_str]

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        if model_args.apply_hyperadapter or model_args.apply_hyperlora:
            data_collator = dialect_data_collator
        else:
            data_collator = label_data_collator
    elif training_args.fp16:
        # NOTE: This might fail with hyperadapter
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if model_args.apply_hyperadapter or model_args.apply_hyperlora:
        optimizer=torch.optim.AdamW([*model.parameters(), *model.hypernet.down_hypernet.parameters(), *model.hypernet.up_hypernet.parameters()],
            lr=training_args.learning_rate)
    elif model_args.apply_adapter:
        optimizer=torch.optim.AdamW([*model.parameters()],
            lr=training_args.learning_rate)
    else:
        optimizer=torch.optim.AdamW([*model.parameters()],
            lr=training_args.learning_rate)

    scheduler=transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=0.06*len(train_dataset)*training_args.num_train_epochs,
        num_training_steps=len(train_dataset)*training_args.num_train_epochs)

    optimizers = optimizer, scheduler

    # Initialize our Trainer
    trainer_class = RobertaTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        dataset_sizes=dataset_sizes,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=optimizers
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(valid_mm_dataset), data_args.max_eval_samples
                )
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics(
                "eval", combined if task is not None and "mnli" in task else metrics
            )

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
