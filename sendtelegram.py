import telegram
TOKEN = '5166911819:AAG5p1UnPGUz4zKROmumi0qdcDEiRmgEml8'

def send_telegram_message(txt):
    bot = telegram.Bot(TOKEN)
    if bot.get_updates():
        chat_id = bot.get_updates()[-1].message.chat_id
        bot.send_message(chat_id,txt)
    else:
        print("Empty list. Please, chat with the bot")

def send_telegram_photo(pic):
    bot = telegram.Bot(TOKEN)
    if bot.get_updates():
        chat_id = bot.get_updates()[-1].message.chat_id
        bot.send_photo(chat_id,photo=open(pic, 'rb'))
    else:
        print("Empty list. Please, chat with the bot")

are='Phát hiện hành vi bất thường'
img_path= "D:\KLTN\\NEW\IMG\DAU LUNG 28_2_2022 18_52_8.jpg"
send_telegram_message(are)
send_telegram_photo(img_path)
