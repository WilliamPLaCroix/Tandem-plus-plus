# Wheeeee~!
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from random import randint
import math
import csv
import pandas as pd
import cefr
import re, string
from tqdm import tqdm
import torch
import numpy as np

pattern = re.compile('[\W_]+')

def calculate_perplexity(predictions, actual_tokens):
    perplexities = []
    for beam in range(10):
        log_probability = 0.0
        num_tokens = len(predictions)

        for i in range(num_tokens):
            token_logits = predictions[i]
            actual_token = actual_tokens[i]
            token_logit = token_logits[beam][actual_token]
            # Calculate the log likelihood of the actual token
            # if torch.isinf(token_logit) == True or token_logit.item() == -1.0000e+09:
            #     log_likelihood += 1
            # else:
            print("token logit:", token_logit.item())
            probability = math.exp(token_logit.item())/(1 + math.exp(token_logit.item()))
            print("probability:", probability)
            #print("probability:", probability)
            if probability != 0:
                log_probability += math.log(probability)
        # Calculate perplexity as the exponential of the negative log-likelihood
        perplexity = math.exp(-log_probability / num_tokens)
        perplexities.append(perplexity)
        # perplexities.append(-log_probability / num_tokens)

    return np.mean(perplexities)

def generate_constraint_list():
    with open('constraint_lists.json', 'r', encoding="utf-8") as f:
        constraint_lists = json.load(f)
        # Picks a random unit (1-12) list from A1 word list 
    unit = randint(1, len(constraint_lists["A1 Full List"]))
    return constraint_lists["A1 Full List"][f"Unit {unit}"]

def generate_constraint_sample(constraint_list):
    list_length = len(constraint_list)
    index = randint(0, list_length-30)
    return constraint_list[index:index+30]

def main():

    # tokenizer = AutoTokenizer.from_pretrained("jphme/Llama-2-13b-chat-german")
    # model = AutoModelForCausalLM.from_pretrained("jphme/Llama-2-13b-chat-german")

    from transformers import AutoTokenizer, AutoModelWithLMHead
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
    model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")

    # Takes a 30-word slice as constraint list for generation
    constraint_list = generate_constraint_list()
    force_words_ids = [
        tokenizer(generate_constraint_sample(constraint_list), add_special_tokens=False).input_ids,
    ]

    # Llama prompt:
    prompt = "[INST] <<SYS>> You are a friendly German language partner. Your role is to chat with language learners to help them improve their language skills. Your answers should be brief, friendly and conversational. Please answer in A1-level German. <</SYS>>"
    # Non-llama prompt:
    # prompt = ["You are a friendly German language partner. Your role is to chat with language learners to help them improve their language skills. Your answers should be brief, friendly and conversational. Please answer in A1-level German."]
    # Most basic prompt:
    # prompt = "Hallo"
    with open('Test set.csv', 'r', encoding="utf-8") as f:
        test_data = list(csv.reader(f, delimiter=","))

    responses = []
    perplexities = []
    grades = []

    run_size = 49
    for question in tqdm(test_data[:run_size]):
        # Llama input
        input = f'{prompt}{question}[/INST]'
        # Non-llama input
        # input = f'{prompt}{question}'
        tokenized_prompt = tokenizer.encode(input, return_tensors="pt")
        # Generate a response 
        print("Generating response...")
        while True:
            try:
                outputs = model.generate(
                    tokenized_prompt,
                    force_words_ids=force_words_ids,
                    num_beams=10,
                    num_return_sequences=1,
                    no_repeat_ngram_size=1,
                    remove_invalid_values=True,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                break
            except ValueError:
                print("ValueError. Restarting current loop with new constraint list...")
                force_words_ids = [
                tokenizer(generate_constraint_sample(constraint_list), add_special_tokens=False).input_ids,
                ]
                pass

        # Extract and print the chatbot's response
        bot_response = tokenizer.decode(outputs.sequences[0])[len(input)+1:]
        responses.append(bot_response)
        print("Chatbot:", bot_response)
        # calculate perplexity
        logits = outputs.scores
        # Calculate perplexity for the model's predictions
        print("logits length:",len(logits))
        print("output sequence:", outputs.sequences[0][-len(logits):])
        print("decoded output:", tokenizer.decode(outputs.sequences[0][-len(logits):]))
        print("sequence length:",len(outputs.sequences[0][-len(logits):])) #[len(tokenized_prompt):]
        perplexity = calculate_perplexity(logits, outputs.sequences[0][-len(logits):])
        perplexities.append(perplexity)
        response = bot_response.split()
        response = [pattern.sub('', word) for word in response]
        grade = cefr.CEFR_level_guesser(response)   
        grades.append(grade)
        print("Perplexity per word:", perplexity)
    
    results_dict = {"input":test_data[:run_size], "response":responses, "perplexity":perplexities, "grade":grades}
    output_data = pd.DataFrame.from_dict(results_dict)
    output_data.to_csv("output_data.csv", encoding="utf-8")

if __name__ == "__main__":    
    main()