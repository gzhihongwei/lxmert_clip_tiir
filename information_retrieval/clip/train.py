import logging
import sys

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    CLIPConfig,
    CLIPProcessor,
    HfArgumentParser,
    set_seed,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from tqdm import tqdm

from information_retrieval.clip import CLIPForIR
from information_retrieval.clip.data import CLIPRetrievalDataset
from information_retrieval.utils import compute_metrics_maker, DataTrainingArguments, ModelArguments


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=Path(sys.argv[1]).resolve())
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"distributed: {training_args.local_rank != -1}, mixed precision: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if Path(training_args.output_dir).is_dir() and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(Path(training_args.output_dir).iterdir()) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = CLIPConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        return_dict=model_args.return_dict,
    )
    processor = CLIPProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    model = CLIPForIR.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    train_dataset = CLIPRetrievalDataset(processor, data_args, "train", is_train=True) if training_args.do_train else None
    val_dataset = CLIPRetrievalDataset(processor, data_args, "minival" if data_args.evaluate_during_training else "val", is_train=False) if training_args.do_eval else None
    test_dataset = CLIPRetrievalDataset(processor, data_args, "test", is_train=False) if training_args.do_predict else None
    
    if not training_args.do_train and not (training_args.do_eval or training_args.do_predict):
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    
    compute_metrics = None
    
    if training_args.do_eval or training_args.do_predict:
        num_captions_per_img = val_dataset.effective_captions_per_img if training_args.do_eval else test_dataset.effective_captions_per_img
        compute_metrics = compute_metrics_maker(num_captions_per_img, data_args.evaluation_output_file)
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else test_dataset if training_args.do_predict else None,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
    if training_args.do_eval or training_args.do_predict:
        logger.info("*********** Evaluate ***********")
        split = "eval" if training_args.do_eval else "test"
        metrics = trainer.evaluate(ignore_keys=model_args.ignore_keys, metric_key_prefix=split)

        trainer.log_metrics(split, metrics)
        trainer.save_metrics(split, metrics)


if __name__ == "__main__":
    main()
