import os

import telebot
from use_tandem import reply, start_model
BOT_TOKEN = '6194987330:AAEpjh7-fArVIvXXTBdDZCG9MGyXBLv53x8'

tokenizer, model, force_words_ids, chat_history=start_model()
bot = telebot.TeleBot(BOT_TOKEN)

max_token_limit = 1000


@bot.message_handler(commands=['start', 'hallo'])
def send_welcome(message):
    bot.reply_to(message, "Hallo, ich bin Tandem++, Ihr Deutsch-Lernpartner")

@bot.message_handler(func=lambda msg: True)
def use_bot(message):
    bot_response, chat_history = reply(message.text,tokenizer, model, force_words_ids, chat_history,max_token_limit)
    bot.reply_to(message, bot_response)
    return chat_history

bot.infinity_polling()
