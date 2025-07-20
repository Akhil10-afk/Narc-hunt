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
    
    try:
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]  # Probability for drug-related class
    except AttributeError:
        pred = model.predict(vec)[0]
        proba = None  # predict_proba not available

    if proba is not None:
        print(f"Text: {text} | Prediction: {pred} | Drug Probability: {proba:.2f}")
    else:
        print(f"Text: {text} | Prediction: {pred} | No probability info available")

    if proba is not None and proba > 0.7:
        bot.reply_to(message, f"🚨 Warning: Drug-related message detected.\nConfidence: {proba:.2f}")
    elif pred == 1:
        bot.reply_to(message, "🚨 Warning: Drug-related message detected.")
    else:
        bot.reply_to(message, "✅ Message looks clean.")

print("🤖 Bot is running...")
bot.polling()
