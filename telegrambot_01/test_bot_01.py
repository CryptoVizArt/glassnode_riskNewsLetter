import os
import yfinance as yf
import matplotlib.pyplot as plt
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

# Function to handle the /announce command to send the Apple stock price chart to the channel
async def announce(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Fetch Apple stock data
    stock_data = yf.download('AAPL', period='1mo', interval='1d')

    # Plot the stock price chart
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label='AAPL Close Price')
    plt.title('Apple (AAPL) Stock Price - Last Month')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True)

    # Save the chart as a PNG image
    chart_filename = 'apple_stock_price.png'
    plt.savefig(chart_filename)
    plt.close()

    # Send the image to the Telegram channel
    with open(chart_filename, 'rb') as chart_file:
        await context.bot.send_photo(chat_id=CHANNEL_ID, photo=chart_file)

    await update.message.reply_text('Apple stock price chart has been sent to the channel!')

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