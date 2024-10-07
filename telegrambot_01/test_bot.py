import time
import yfinance as yf
import matplotlib.pyplot as plt
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext
import os
import asyncio

# Telegram Bot credentials
TELEGRAM_BOT_TOKEN = '7602146484:AAFfBSxHpqljMohWJ9b9O826iN_0PBjrDm0'
TELEGRAM_CHAT_ID = ''

# Initialize the Telegram bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)


def fetch_and_plot_stock():
    # Fetch Apple stock data
    stock_data = yf.download('AAPL', period='1d', interval='2m')

    # Plot the stock price
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.index, stock_data['Close'], label='AAPL Close Price', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.title('AAPL Stock Price')
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig('apple_stock.png')
    plt.close()


async def send_stock_chart():
    # Send the saved chart to the Telegram channel
    await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=open('apple_stock.png', 'rb'))


def get_updates():
    updates = bot.get_updates()
    for update in updates:
        print(update)

get_updates()

def main():
    while True:
        try:
            fetch_and_plot_stock()
            asyncio.run(send_stock_chart())
        except Exception as e:
            print(f"Error occurred: {e}")

        # Wait for 2 minutes before the next update
        time.sleep(120)


if __name__ == '__main__':
    main()