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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load environment variables from the .env file
load_dotenv()

# Get the bot token and channel ID from environment variables
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')
API_KEY = os.getenv('GLASSNODE_API_KEY')

# Constants for Glassnode API
SINCE_DATE = int((datetime.now() - timedelta(days=730)).timestamp())  # Last 2 years
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


def calculate_momentum_rsi(df, column='price_usd_close', rsi_window=14, window_norm=90, normalize=True):
    price_change = df[column].diff()
    gains = price_change.where(price_change > 0, 0)
    losses = -price_change.where(price_change < 0, 0)
    avg_gains = gains.rolling(window=rsi_window, min_periods=1).mean()
    avg_losses = losses.rolling(window=rsi_window, min_periods=1).mean()
    relative_strength = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + relative_strength))

    if normalize:
        rsi_min = rsi.rolling(window=window_norm, min_periods=1).min()
        rsi_max = rsi.rolling(window=window_norm, min_periods=1).max()
        normalized_momentum = 2 * (rsi - rsi_min) / (rsi_max - rsi_min) - 1
        return normalized_momentum
    else:
        return rsi


def calculate_spot_cvd_bias(df, column='spot_cvd_sum', window_sum=7, window_norm=90, normalize=True):
    rolling_sum = df[column].rolling(window=window_sum).sum()

    if normalize:
        rolling_min = rolling_sum.rolling(window=window_norm, min_periods=1).min()
        rolling_max = rolling_sum.rolling(window=window_norm, min_periods=1).max()
        normalized_bias = 2 * (rolling_sum - rolling_min) / (rolling_max - rolling_min) - 1
        return normalized_bias
    else:
        return rolling_sum


def calculate_spot_volume_momentum(df, column='spot_volume_daily_sum', fast_window=7, slow_window=90, window_norm=90,
                                   normalize=True):
    fast_ma = df[column].rolling(window=fast_window).mean()
    slow_ma = df[column].rolling(window=slow_window).mean()
    volume_momentum = fast_ma / slow_ma

    if normalize:
        rolling_min = volume_momentum.rolling(window=window_norm, min_periods=1).min()
        rolling_max = volume_momentum.rolling(window=window_norm, min_periods=1).max()
        normalized_momentum = 2 * (volume_momentum - rolling_min) / (rolling_max - rolling_min) - 1
        return normalized_momentum
    else:
        return volume_momentum


def aggregate_indicators(df, columns, method='equal_weight'):
    if not all(col in df.columns for col in columns):
        raise ValueError("Some specified columns are not in the DataFrame")

    data = df[columns]

    if method == 'equal_weight':
        weight = 1 / len(columns)
        print(f"weights are all equal = {weight:.4f}")
        return data.mean(axis=1)

    elif method == 'PCA':
        data_clean = data.dropna()

        if len(data_clean) == 0:
            raise ValueError("No complete rows found after removing NaN values")

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)

        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(data_scaled)

        weights = pca.components_[0] / np.sum(np.abs(pca.components_[0]))

        print("PCA weights:")
        for col, weight in zip(columns, weights):
            print(f"{col}: {weight:.4f}")

        weighted_sum = data.mul(weights).sum(axis=1)

        min_val, max_val = data.min().min(), data.max().max()
        normalized_sum = (weighted_sum - weighted_sum.min()) / (weighted_sum.max() - weighted_sum.min())
        normalized_sum = normalized_sum * (max_val - min_val) + min_val

        return normalized_sum

    else:
        raise ValueError("Invalid method. Choose 'equal_weight' or 'PCA'")


def create_indicator_chart(merged_df, indicator_column, chart_title):
    one_year_ago = datetime.now() - timedelta(days=730)
    merged_df_last_year = merged_df[merged_df.index > one_year_ago]

    GREY_COLOR = 'rgba(128, 128, 128, 0.7)'

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=merged_df_last_year.index, y=merged_df_last_year['price_usd_close'], name="Price USD",
                   line=dict(color=GREY_COLOR, width=2), mode='lines'),
        secondary_y=False,
    )

    indicator = merged_df_last_year[indicator_column]
    fig.add_trace(
        go.Scatter(
            x=merged_df_last_year.index,
            y=indicator,
            name=indicator_column,
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)',
            mode='lines'
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df_last_year.index,
            y=indicator.where(indicator < 0, 0),
            name=f"{indicator_column} (Negative)",
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)',
            mode='lines'
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df_last_year.index,
            y=[0] * len(merged_df_last_year),
            showlegend=False,
            line=dict(color=GREY_COLOR, width=2),
            hoverinfo='skip'
        ),
        secondary_y=True,
    )

    for month in [1, 4, 7, 10]:
        for year in range(merged_df_last_year.index[0].year, merged_df_last_year.index[-1].year + 1):
            date = pd.Timestamp(year=year, month=month, day=1)
            if merged_df_last_year.index[0] <= date <= merged_df_last_year.index[-1]:
                fig.add_vline(x=date, line_dash="dash", line_color=GREY_COLOR, line_width=0.75, opacity=0.7)

    last_value = indicator.iloc[-1]
    last_date = indicator.index[-1]

    indicator_color = 'green' if last_value >= 0 else 'red'

    fig.add_annotation(
        x=0.95,
        y=last_value * 0.88,
        xref="paper",
        yref="y2",
        text=f"{last_value:.2f}",
        showarrow=False,
        font=dict(size=18, color=indicator_color),
        align="left",
        xanchor="left",
        yanchor="middle",
    )

    fig.update_layout(
        title={
            'text': chart_title,
            'font': {'color': 'black', 'size': 18, 'weight': 'bold'}
        },
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': 'black', 'size': 14},
        width=1000,
        height=450,
    )

    fig.update_xaxes(
        showgrid=False,
        tickfont={'color': 'black', 'size': 14},
        zeroline=False,
        title_text=''
    )
    fig.update_yaxes(
        showgrid=False,
        secondary_y=False,
        tickfont={'color': GREY_COLOR, 'size': 14},
        zeroline=False,
        showline=True,
        linecolor=GREY_COLOR,
        ticks='outside',
        tickcolor=GREY_COLOR,
        title_text='',
        title_font=dict(size=18)
    )
    fig.update_yaxes(
        showgrid=False,
        secondary_y=True,
        range=[-1, 1],
        tickfont={'color': GREY_COLOR, 'size': 14},
        zeroline=False,
        showline=True,
        linecolor=GREY_COLOR,
        ticks='outside',
        side='right',
        tickcolor=GREY_COLOR,
        title_text='',
        title_font=dict(size=18)
    )

    return fig


def create_aggregator_chart(merged_df, indicator_column, chart_title):
    one_year_ago = datetime.now() - timedelta(days=730)
    merged_df_last_year = merged_df[merged_df.index > one_year_ago]

    GREY_COLOR = 'rgba(128, 128, 128, 0.7)'
    RED_COLOR = 'red'
    BLUE_COLOR = 'blue'

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=merged_df_last_year.index, y=merged_df_last_year['price_usd_close'], name="Price USD",
                   line=dict(color=GREY_COLOR, width=2), mode='lines'),
        secondary_y=False,
    )

    indicator = merged_df_last_year[indicator_column]

    fig.add_trace(
        go.Scatter(
            x=merged_df_last_year.index,
            y=indicator.where(indicator > 0, 0),
            name=f"{indicator_column} (Positive)",
            line=dict(color=RED_COLOR, width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0)',
            mode='lines'
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df_last_year.index,
            y=indicator.where(indicator < 0, 0),
            name=f"{indicator_column} (Negative)",
            line=dict(color=BLUE_COLOR, width=2),
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0)',
            mode='lines'
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df_last_year.index,
            y=[0] * len(merged_df_last_year),
            showlegend=False,
            line=dict(color=GREY_COLOR, width=2),
            hoverinfo='skip'
        ),
        secondary_y=True,
    )

    for month in [1, 4, 7, 10]:
        for year in range(merged_df_last_year.index[0].year, merged_df_last_year.index[-1].year + 1):
            date = pd.Timestamp(year=year, month=month, day=1)
            if merged_df_last_year.index[0] <= date <= merged_df_last_year.index[-1]:
                fig.add_vline(x=date, line_dash="dash", line_color=GREY_COLOR, line_width=0.75, opacity=0.7)

    last_value = indicator.iloc[-1]

    indicator_color = RED_COLOR if last_value >= 0 else BLUE_COLOR

    fig.add_annotation(
        x=0.95,
        y=last_value * 0.88,
        xref="paper",
        yref="y2",
        text=f"{last_value:.2f}",
        showarrow=False,
        font=dict(size=18, color=indicator_color),
        align="left",
        xanchor="left",
        yanchor="middle",
    )

    fig.update_layout(
        title={
            'text': chart_title,
            'font': {'color': 'black', 'size': 18, 'weight': 'bold'}
        },
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': 'black', 'size': 14},
        width=1000,
        height=450,
    )

    fig.update_xaxes(
        showgrid=False,
        tickfont={'color': 'black', 'size': 14},
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=False,
        secondary_y=False,
        tickfont={'color': GREY_COLOR, 'size': 14},
        zeroline=False,
        showline=True,
        linecolor=GREY_COLOR,
        ticks='outside',
        tickcolor=GREY_COLOR,
        title_text='',
        title_font=dict(size=18)
    )
    fig.update_yaxes(
        showgrid=False,
        secondary_y=True,
        range=[-1, 1],
        tickfont={'color': GREY_COLOR, 'size': 14},
        zeroline=False,
        showline=True,
        linecolor=GREY_COLOR,
        ticks='outside',
        side='right',
        tickcolor=GREY_COLOR,
        title_text='',
        title_font=dict(size=18)
    )

    return fig


def chart_aggregate(df, indicator_col, price_col, chart_title="Price and Indicator Chart"):
    last_2_years = df.index.max() - pd.DateOffset(years=2)
    df_last_2_years = df[df.index >= last_2_years]

    GREY_COLOR = 'rgba(128, 128, 128, 0.7)'

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_last_2_years.index,
            y=df_last_2_years[price_col],
            mode='markers',
            marker=dict(
                color=df_last_2_years[indicator_col],
                size=8,
                line=dict(width=0.5, color='black'),
                colorscale='RdYlGn_r',
                colorbar=dict(
                    title='Indicator',
                    tickvals=[-1, 0, 1],
                    ticktext=['-1', '0', '1'],
                    lenmode='fraction',
                    len=0.9,
                    thickness=20,
                    x=1.02,
                    xpad=10,
                ),
                cmin=-1,
                cmax=1,
            ),
            text=[f"Date: {date.strftime('%Y-%m-%d')}<br>Price: {price:.2f}<br>Indicator: {ind:.2f}"
                  for date, price, ind in zip(df_last_2_years.index, df_last_2_years[price_col], df_last_2_years[indicator_col])],
            hoverinfo='text'
        )
    )

    start_date = df_last_2_years.index[0].to_period('Q').to_timestamp()
    end_date = df_last_2_years.index[-1].to_period('Q').to_timestamp()
    for date in pd.date_range(start=start_date, end=end_date, freq='Q'):
        if df_last_2_years.index[0] <= date <= df_last_2_years.index[-1]:
            fig.add_vline(x=date, line_dash="dash", line_color=GREY_COLOR, line_width=1, opacity=0.7)

    last_value = df_last_2_years[indicator_col].iloc[-1]
    last_price = df_last_2_years[price_col].iloc[-1]

    fig.add_annotation(
        x=df_last_2_years.index[-1],
        y=last_price,
        text=f"{last_value:.2f}",
        showarrow=False,
        font=dict(size=14, color='red' if last_value > 0 else 'green'),
        align="left",
        xanchor="left",
        yanchor="middle",
        xshift=10,
    )

    fig.update_layout(
        title={
            'text': chart_title,
            'font': {'color': 'black', 'size': 18, 'weight': 'bold'}
        },
        showlegend=False,
        hovermode="closest",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': 'black', 'size': 14},
        width=1000,
        height=450,
        margin=dict(l=40, r=120, t=60, b=40)
    )

    fig.update_xaxes(
        showgrid=False,
        tickfont={'color': 'black', 'size': 12},
        zeroline=False,
        dtick="M3",
        tickformat="%b %Y",
        showline=False,
        mirror=False
    )
    fig.update_yaxes(
        showgrid=False,
        tickfont={'color': GREY_COLOR, 'size': 12},
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=GREY_COLOR,
        mirror=False,
        ticks='outside',
        ticklen=5,
        tickcolor=GREY_COLOR
    )

    return fig

def create_bar_chart(merged_df, column):
    def calculate_periodic_changes(df, column):
        weekly_change = df[column].pct_change(7).iloc[-1] * 100
        monthly_change = df[column].pct_change(30).iloc[-1] * 100
        quarterly_change = df[column].pct_change(90).iloc[-1] * 100
        return weekly_change, monthly_change, quarterly_change

    GREY_COLOR = 'rgba(128, 128, 128, 0.7)'
    RED_COLOR = 'red'
    BLUE_COLOR = 'blue'

    weekly_change, monthly_change, quarterly_change = calculate_periodic_changes(merged_df, column)
    periods = ['Quarterly', 'Monthly', 'Weekly']
    changes = [quarterly_change, monthly_change, weekly_change]
    colors = [RED_COLOR if change > 0 else BLUE_COLOR for change in changes]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=periods,
            y=changes,
            marker_color=colors,
            text=[f"{change:.2f}%" for change in changes],
            textposition='outside',
            width=0.5
        )
    )

    fig.update_layout(
        title={
            'text': f"Rate of Change {column}",
            'font': {'color': 'black', 'size': 18, 'weight': 'bold'}
        },
        xaxis_title=None,
        yaxis_title="Rate of Change",
        yaxis_ticksuffix="%",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': 'black', 'size': 14},
        width=600,
        height=600,
        hovermode="x unified",
    )

    fig.add_hline(y=0, line_dash="solid", line_color=GREY_COLOR, line_width=1)

    fig.update_yaxes(
        showgrid=False,
        tickfont={'color': GREY_COLOR, 'size': 14},
        zeroline=False,
        linecolor=GREY_COLOR,
        showline=True,
        ticks='outside',
        tickcolor=GREY_COLOR,
        dtick=40
    )

    fig.update_xaxes(
        showgrid=False,
        tickfont={'color': GREY_COLOR, 'size': 14},
        zeroline=False,
        linecolor=None,
        showline=False,
        ticks='outside',
        tickcolor=GREY_COLOR
    )

    return fig

def create_bitcoin_charts():
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

    # Apply the functions to our merged_df
    merged_df['Price Momentum'] = calculate_momentum_rsi(merged_df, column='price_usd_close', rsi_window=14, window_norm=90, normalize=True)
    merged_df['Spot CVD Bias'] = calculate_spot_cvd_bias(merged_df, column='spot_cvd_sum', window_sum=7, window_norm=90, normalize=True)
    merged_df['Spot Volume Momentum'] = calculate_spot_volume_momentum(merged_df, column='spot_volume_daily_sum', fast_window=7, slow_window=90, window_norm=90, normalize=True)

    merged_df['aggregate_indicator_equalweight'] = aggregate_indicators(merged_df, columns=['Price Momentum','Spot CVD Bias','Spot Volume Momentum'], method='equal_weight')
    merged_df['aggregate_indicator_PCA'] = aggregate_indicators(merged_df, columns=['Price Momentum','Spot CVD Bias','Spot Volume Momentum'], method='PCA')

    charts = []

    # Create charts for each indicator
    indicators = [
        ('Price Momentum', "Bitcoin: Price Momentum"),
        ('Spot CVD Bias', "Bitcoin: Spot CVD Bias"),
        ('Spot Volume Momentum', "Bitcoin: Spot Volume Momentum")
    ]

    for indicator, title in indicators:
        fig = create_indicator_chart(merged_df, indicator, title)
        charts.append((fig, f'bitcoin_analysis_{indicator}_last_year.png'))

    # Create charts for each aggregator
    aggregators = [
        ('aggregate_indicator_equalweight', "Bitcoin: Spot Aggregate Indicator EW"),
        ('aggregate_indicator_PCA', "Bitcoin: Spot Aggregate Indicator PCA")
    ]

    for aggregator, title in aggregators:
        fig = create_aggregator_chart(merged_df, aggregator, title)
        charts.append((fig, f'bitcoin_analysis_spot_{aggregator}_one_year.png'))

    # Create scatter plot
    fig = chart_aggregate(merged_df, 'aggregate_indicator_equalweight', 'price_usd_close', "Spot Sentiment Aggregated Indicator")
    charts.append((fig, 'bitcoin_analysis_spot_aggr_last_1year.png'))

    # Create bar charts
    for column in ['aggregate_indicator_equalweight', 'aggregate_indicator_PCA']:
        fig = create_bar_chart(merged_df, column)
        charts.append((fig, f'{column}_periodic_changes_chart.png'))

    # Save all charts as PNG files
    for fig, filename in charts:
        pio.write_image(fig, filename)

    return charts

# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! Welcome to the Bitcoin Analysis Bot. Use /announce to get the latest charts.')

# Function to handle the /announce command
async def announce(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        # Create the Bitcoin analysis charts
        charts = create_bitcoin_charts()

        # Send all charts to the Telegram channel
        for _, filename in charts:
            with open(filename, 'rb') as chart_file:
                await context.bot.send_photo(chat_id=CHANNEL_ID, photo=chart_file)

        await update.message.reply_text('All Bitcoin analysis charts have been sent to the channel!')
    except Exception as e:
        error_message = f"An error occurred while creating or sending the charts: {str(e)}"
        print(error_message)  # Log the error
        await update.message.reply_text(f"Sorry, an error occurred: {error_message}")

def main():
    # Create an Application object using the bot token
    application = ApplicationBuilder().token(TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("announceSPOT", announce))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()

