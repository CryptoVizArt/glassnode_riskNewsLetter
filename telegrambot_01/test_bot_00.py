import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables from the .env file
load_dotenv()

# Get the bot token from the environment variable
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Replace with your channel username or ID
CHANNEL_ID = '@testChannel_zi'  # or '-1001234567890' if using channel ID

# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! Welcome to the bot. How can I help you today?')

# Function to handle messages that contain "hello"
async def respond_hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text.lower()
    if user_message == 'hello':
        await update.message.reply_text('Hello!')

# Function to handle the /announce command to send a message to the channel
async def announce(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_to_send = 'This is an announcement to the channel!'
    await context.bot.send_message(chat_id=CHANNEL_ID, text=message_to_send)
    await update.message.reply_text('Announcement sent to the channel!')

def main():
    # Create an Application object using the bot token
    application = ApplicationBuilder().token(TOKEN).build()

    # Add a command handler for the /start command
    application.add_handler(CommandHandler("start", start))

    # Add a message handler to respond to the message "hello"
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond_hello))

    # Add a command handler for the /announce command
    application.add_handler(CommandHandler("announce", announce))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()