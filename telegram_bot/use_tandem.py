# Wheeeee~!
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from random import randint

def create_constraints_list(tokenizer):
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
    return force_words_ids
def handle_history(chat_history, user_input,tokenizer,max_token_limit):
    # Add user input to the chat history
    chat_history.append(user_input + "\n")

    # Concatenate and truncate the chat history to fit within the token limit
    combined_history = " ".join(chat_history)
    combined_history += " [/INST]"
    context = combined_history
    print("history so far:", combined_history)
    if len(tokenizer.encode(combined_history)) > max_token_limit:
        # Truncate the chat history to fit within the token limit
        while len(tokenizer.encode(combined_history)) > max_token_limit:
            chat_history.pop(1)

    tokenized_prompt = tokenizer.encode(combined_history, return_tensors="pt")
    return tokenized_prompt,context

def start_model():
    tokenizer = AutoTokenizer.from_pretrained("jphme/Llama-2-13b-chat-german")
    model = AutoModelForCausalLM.from_pretrained("jphme/Llama-2-13b-chat-german")


    force_words_ids = create_constraints_list(tokenizer)
    chat_history = ["[INST] <<SYS>> You are a friendly German language partner. Your role is to chat with language learners to help them improve their language skills. Your answers should be brief, friendly and conversational. Please answer in A1-level German. <</SYS>>\n"]

    return tokenizer, model, force_words_ids, chat_history

def reply(user_input,tokenizer, model, force_words_ids, chat_history,max_token_limit):
        
        tokenized_prompt, context = handle_history(chat_history, user_input,tokenizer,max_token_limit)
        
        # Generate a response based on the chat history
        outputs = model.generate(
            tokenized_prompt,
            force_words_ids=force_words_ids,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
            max_new_tokens=50,
        )

        # Extract and print the chatbot's response
        bot_response = tokenizer.decode(outputs[0])[len(context):]
        print("Chatbot: ", bot_response)

        # Add the bot's response to the chat history
        chat_history.append(bot_response + "\n")
        return bot_response,chat_history
