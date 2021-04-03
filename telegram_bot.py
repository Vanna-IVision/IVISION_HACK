import telebot
import time
import json
bot = telebot.TeleBot('1786817109:AAFPNlOAhqYDeL1q6ymUp0G80WKUbwhT-f0')
keyboard1 = telebot.types.ReplyKeyboardMarkup()
keyboard1.row('Парковка ул. Ленина')
sleep_time = 30
from utils import get_free_from_json


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, ты написал мне /start', reply_markup=keyboard1)

@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text.lower() == 'парковка ул. ленина':
        prev_value = -1
        while(True):
            value = int(get_free_from_json())
            if(value==prev_value):
                time.sleep(sleep_time)
                continue
            if(prev_value==-1):
                bot.send_message(message.chat.id, "свободно {} мест".format(value))
                image = open("image_1.jpg", 'rb')
                bot.send_photo(message.chat.id, image)
            else:
                cur = value - prev_value
                if cur>0:
                    bot.send_message(message.chat.id, "освободилось {} мест(о/а)".format(cur))
                    image = open("image_1.jpg", 'rb')
                    bot.send_photo(message.chat.id, image)
                else:
                    bot.send_message(message.chat.id, "стало занято {} мест(о/а)".format(abs(cur)))
                    image = open("image_1.jpg", 'rb')
                    bot.send_photo(message.chat.id, image)
            prev_value = value
            time.sleep(sleep_time)


bot.polling()


