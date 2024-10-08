import os
import requests
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables from the .env file
load_dotenv()

# Get the bot token and channel ID from environment variables
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')
API_KEY = os.getenv('GLASSNODE_API_KEY')

# Constants for Glassnode API
SINCE_DATE = int((datetime.now() - timedelta(days=365)).timestamp())  # Last 1 year
UNTIL_DATE = int(datetime.now().timestamp())

# URLs for fetching data
PRICE_URL = 'https://api.glassnode.com/v1/metrics/market/price_usd_close'
METRICS = [
    'https://api.glassnode.com/v1/metrics/market/spot_cvd_sum',
    'https://api.glassnode.com/v1/metrics/market/spot_volume_daily_sum'
]

def fetch_glassnode_data(url, asset='BTC'):
    params = {
        'a': asset,
        's': SINCE_DATE,
        'u': UNTIL_DATE,
        'api_key': API_KEY,
        'f': 'CSV',
        'c': 'USD'
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text))
        metric_name = url.split('/')[-1]
        df.columns = ['t', metric_name]
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')
        return df
    else:
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return None

def create_bitcoin_chart():
    # Fetch and merge data
    price_df = fetch_glassnode_data(PRICE_URL)
    all_dfs = [price_df]
    for metric_url in METRICS:
        metric_df = fetch_glassnode_data(metric_url)
        if metric_df is not None:
            all_dfs.append(metric_df)

    merged_df = pd.concat(all_dfs, axis=1)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    merged_df.set_index('t', inplace=True)

    # Create the chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=('BTC Price', 'Spot CVD Sum', 'Spot Volume Daily Sum'))

    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['price_usd_close'], name='BTC Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['spot_cvd_sum'], name='Spot CVD Sum'), row=2, col=1)
    fig.add_trace(go.Bar(x=merged_df.index, y=merged_df['spot_volume_daily_sum'], name='Spot Volume Daily Sum'), row=3, col=1)

    fig.update_layout(height=900, width=1200, title_text="Bitcoin Analysis - Last 1 Year")
    fig.update_xaxes(rangeslider_visible=False)

    # Save the chart as a PNG image
    pio.write_image(fig, 'bitcoin_analysis.png')

# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! Welcome to the Bitcoin Analysis Bot. Use /announce to get the latest chart.')

# Function to handle the /announce command
async def announce(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        # Create the Bitcoin analysis chart
        create_bitcoin_chart()

        # Send the image to the Telegram channel
        with open('bitcoin_analysis.png', 'rb') as chart_file:
            await context.bot.send_photo(chat_id=CHANNEL_ID, photo=chart_file)

        await update.message.reply_text('Bitcoin analysis chart has been sent to the channel!')
    except Exception as e:
        error_message = f"An error occurred while creating or sending the chart: {str(e)}"
        print(error_message)  # Log the error
        await update.message.reply_text(f"Sorry, an error occurred: {error_message}")

def main():
    # Create an Application object using the bot token
    application = ApplicationBuilder().token(TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("announce", announce))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()