from flask import Flask, request
import telebot
import joblib
import os

API_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
bot = telebot.TeleBot(API_TOKEN)
app = Flask(__name__)

# Load your model
model = joblib.load("ai-modules/model/model.pkl")
vectorizer = joblib.load("ai-modules/model/vectorizer.pkl")

# Webhook route
@app.route(f"/{API_TOKEN}", methods=['POST'])
def webhook():
    json_str = request.get_data().decode('utf-8')
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return "!", 200

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text
    vec = vectorizer.transform([text])
    try:
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]
    except AttributeError:
        pred = model.predict(vec)[0]
        proba = None

    if proba is not None and proba > 0.7:
        bot.reply_to(message, f"🚨 Warning: Drug-related message detected.\nConfidence: {proba:.2f}")
    elif pred == 1:
        bot.reply_to(message, "🚨 Warning: Drug-related message detected.")
    else:
        bot.reply_to(message, "✅ Message looks clean.")

# Setup webhook when server starts
@app.before_first_request
def setup_webhook():
    webhook_url = f"https://<your-render-url>.onrender.com/{API_TOKEN}"
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
