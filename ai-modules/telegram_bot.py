import telebot
import joblib

# Load model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Replace with your actual bot token
BOT_TOKEN = "7702451388:AAHwfqzZ9n43oyE3QtjGzi9nYjs94vS6uQU"
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    
    if pred[0] == 1:
        bot.reply_to(message, "🚨 Warning: Drug-related message detected.")
    else:
        bot.reply_to(message, "✅ Message looks clean.")

print("🤖 Bot is running...")
bot.polling()
