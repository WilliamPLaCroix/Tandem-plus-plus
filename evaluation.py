# Wheeeee~!
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from random import randint
import math
import csv
import pandas as pd
def calculate_perplexity(predictions, actual_tokens):
    log_likelihood = 0.0
    num_tokens = len(actual_tokens)

    for i in range(num_tokens):
        token_probabilities = predictions[i]
        actual_token = actual_tokens[i]
        token_probability = token_probabilities[actual_token]

        # Calculate the log likelihood of the actual token
        log_likelihood += -math.log(token_probability)

    # Calculate perplexity as the exponential of the negative log-likelihood
    perplexity = math.exp(log_likelihood / num_tokens)
    return perplexity

def main():

    tokenizer = AutoTokenizer.from_pretrained("jphme/Llama-2-13b-chat-german")
    model = AutoModelForCausalLM.from_pretrained("jphme/Llama-2-13b-chat-german")

    with open('constraint_lists.json', 'r', encoding="utf-8") as f:
        constraint_lists = json.load(f)
    # Picks a random unit (1-12) list from A1 word list 
    unit = randint(1, len(constraint_lists["A1 Full List"]))
    constraint_list = constraint_lists["A1 Full List"][f"Unit {unit}"]
    # Takes a 30-word slice as constraint list for generation
    list_length = len(constraint_list)
    index = randint(0, list_length-20)
    constraint_sample = constraint_list[index:index+20]
    force_words_ids = [
        tokenizer(constraint_sample, add_special_tokens=False).input_ids,
    ]

    max_token_limit = 1000

    prompt = ["[INST] <<SYS>> You are a friendly German language partner. Your role is to chat with language learners to help them improve their language skills. Your answers should be brief, friendly and conversational. Please answer in A1-level German. <</SYS>>\n"]
    with open('Test set.csv', 'r') as f:
        test_data = list(csv.reader(f, delimiter=","))

    responses = []
    perplexities = []
    for input in test_data:
        input = prompt + '\n' + input[0] + '[/INST]'
        tokenized_prompt = tokenizer.encode(input, return_tensors="pt")
        # Generate a response 
        outputs = model.generate(
            tokenized_prompt,
            force_words_ids=force_words_ids,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
            max_new_tokens=50,
            output_scores= True
        )

        # Extract and print the chatbot's response
        bot_response = tokenizer.decode(outputs[0])[len(input):]
        responses.append(bot_response)

        print("Chatbot: ", bot_response)
        # calculate perplexity
        logits = outputs[1]
        # Calculate perplexity for the model's predictions
        perplexity = calculate_perplexity(logits, tokenized_prompt)
        perplexities.append(perplexity)
    
    results_dict = {"input":test_data, "response":responses, "perplexity":perplexities}
    pd.Dataframe(results_dict).to_csv("evaulation_output.csv")

        

        
main()