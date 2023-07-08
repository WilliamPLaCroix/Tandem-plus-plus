import argparse
import datetime
import os
import re
import logging
import math
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers
from tqdm import tqdm, trange

parser = argparse.ArgumentParser(description="Baseline chatbot")
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--output_dir", default="model", type=int, help="Output directory.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
transformers.utils.check_min_version("4.17.0.dev0")
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(transformers.MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
        

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


class BaselineChatbotModel():
    """
    Baseline chatbot model with finetuned GPT-2
    """

    def __init__(self, model_path: str, tokenizer_name: str, model_name: str="model") -> None:
        """
        :param str model_path: path to a model file
        :param str tokenizer_name: name of a tokenizer, which is used in reply function
        :param str model_name: model name that is used in self.take_conversations() function, defaults to "model"
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.__model = transformers.AutoModelWithLMHead.from_pretrained(self.model_path)
        self.device = tf.device("cuda" if tf.cuda.is_available() else "cpu")
        self.__model.to(self.device)
        self.model_name = model_name

    @property
    def tokenizer(self) -> object:
        return self.__tokenizer

    @property
    def model(self) -> object:
        return self.__model
    
    def delete_unnecessary_words(self, message: str) -> str:
        """Delete unnecessary words.

        :param str message: original message, which is outputted from this model
        :return str: message, which is modified through this function
        """
        processed_message = message.split('[SEP]</s>')[1]

        processed_message = re.sub(r"\n", "", processed_message)

        processed_message = processed_message.replace('</s>', '')
        processed_message = processed_message.replace('/', '')

        processed_message = re.sub(r"\[*\]", "", processed_message)
        
        processed_message = processed_message + "\n"

        return processed_message

    def reply(self, input_message: str, max_length: int=128) -> str:
        """Reply to the input message.
        
        :param str input_message: message, which is inputted to this model
        :param str max_length: maximum length of the output message (The length is about token not letter or word.)
        :return str: message, which is outputted from this model
        """
        actual_input_message = "<s>" + str(input_message) + "[SEP]"
        input_vector = self.tokenizer.encode(actual_input_message, return_tensors='pt').to(self.device)

        output = self.model.generate(input_vector, do_sample=True, max_length=128, num_return_sequences=1,
                                     pad_token_id=2, top_p=0.95, top_k=50, bad_words_ids=[[1], [5]], no_repeat_ngram_size=3)

        actual_response = ""
        for response in self.tokenizer.batch_decode(output):
            actual_response += self.delete_unnecessary_words(response)

        return actual_response

    def take_conversations(self, num: int=None) -> None:
        """Talk with this model.

        :param int num: the number of conversations (If this is None, the conversation will not end automatically.), defaults to None
        """
        count = 0

        while True:
            print("You: ")
            input_message = input()
            output_message = self.reply(input_message)
            print(f"{self.model_name}: ")
            print(f"{output_message}")

            if count == num:
                break
            else:
                count += 1

    # def train(self) -> None:
    #     checkpoint = None
    #     if args.resume_from_checkpoint is not None:
    #         checkpoint = args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()  # Saves the tokenizer too for easy upload

    #     metrics = train_result.metrics

    #     # max_train_samples = (
    #     #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    #     # )
    #     metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load the data
    train_dataset = np.loadtxt("gutenberg_dataset_de/turn_pairs_train.txt", dtype=int)
    eval_dataset = np.loadtxt("gutenberg_dataset_de/turn_pairs_dev.txt", dtype=int)

    # TODO: Create the model and train it
    model = BaselineChatbotModel(model_path="dbmdz/german-gpt2",
                                 tokenizer_name="dbmdz/german-gpt2",
                                 model_name="baseline_chatbot_model")
    
    # Initialize our Trainer
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=model.__tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        # data_collator=default_data_collator,
        # compute_metrics=compute_metrics if training_args.do_eval else None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )
    trainer.train()

    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # model.train(dataset=train_dataset)
    # model.take_conversations()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)