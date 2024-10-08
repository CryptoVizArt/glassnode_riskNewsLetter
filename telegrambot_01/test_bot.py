from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Replace 'YOUR_NEW_BOT_TOKEN' with your actual bot token
TOKEN = '7602146484:AAFfBSxHpqljMohWJ9b9O826iN_0PBjrDm0'

# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! Welcome to the bot. How can I help you today?')

# Function to handle messages that contain "hello"
async def respond_hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text.lower()
    if user_message == 'hello':
        await update.message.reply_text('Hello!')

def main():
    # Create an Application object using the bot token
    application = ApplicationBuilder().token(TOKEN).build()

    # Add a command handler for the /start command
    application.add_handler(CommandHandler("start", start))

    # Add a message handler to respond to the message "hello"
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond_hello))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()