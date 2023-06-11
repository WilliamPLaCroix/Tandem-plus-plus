import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers

parser = argparse.ArgumentParser(description="Baseline chatbot")
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class BaselineChatbotModel():
    """
    Baseline chatbot model with GPT-2
    """

    def __init__(self, model_path: str, tokenizer_name: str, model_name: str="model") -> None:
        """
        :param str model_path: path to a model file
        :param str tokenizer_name: name of a tokenizer, which is used in reply function
        :param str model_name: model name that is used in self.take_conversations() function, defaults to "model"
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.__tokenizer = transformers.AutoTokenizerAutoTokenizer.from_pretrained(self.tokenizer_name)
        self.__model = transformers.AutoModelWithLMHeadAutoModelForCausalLM.from_pretrained(self.model_path)
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

    # Load the data
    dataset = np.loadtxt("gutenberg_dataset_de/turn_pairs_train.txt", dtype=int)

    # TODO: Create the model and train it
    model = BaselineChatbotModel(model_path="dbmdz/german-gpt2",
                                 tokenizer_name="dbmdz/german-gpt2",
                                 model_name="baseline_chatbot_model")

    model.take_conversations()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)