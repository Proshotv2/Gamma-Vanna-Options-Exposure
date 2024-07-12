import sys
import json
import requests
from datetime import datetime, timedelta, time as dt_time
from py_vollib_vectorized import price_dataframe, get_all_greeks, vectorized_implied_volatility
import numpy as np
import pytz
from scipy.stats import norm
import math
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


app = dash.Dash(__name__, title="MoonShotFlows",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        ],)

app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=120 * 1000, n_intervals=0),  # 2 minutes interval

    # Top row with Snapshot and Gauge
    html.Div([
        # Snapshot column
        html.Div([
            html.H2("Snapshot"),
            html.Div(id='output-container', style={'padding': '10px'}),
        ], style={'width': '50%', 'display': 'inline-block'}),

        # Gauge column
        html.Div([
            dcc.Graph(id='gauge-chart'),
        ], style={'width': '50%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'width': '100%'}),

    # Second row with histograms
    html.Div([
        dcc.Graph(id='gex-histogram'),
    ], style={'width': '100%', 'padding': '10px'}),

    html.Div([
        dcc.Graph(id='vex-histogram'),
    ], style={'width': '100%', 'padding': '10px'}),

    html.Div([
        dcc.Graph(id='dex-histogram'),
    ], style={'width': '100%', 'padding': '10px'}),

    # Accordion buttons and content sections
    html.Details([
        html.Summary('State 1: Positive Gamma/Positive Vanna'),
        html.Div(
            'For every 1% move up (Every Strike) expect selling to increase. For every 1% move down expect buying to increase. If we are in between green strikes identify largest strikes to stay between. Below the flip point price will bounce first positive strike.',
            style={'padding': '20px'})
    ]),
    html.Details([
        html.Summary('State 2: Positive Gamma/Negative Vanna'),
        html.Div(
            'For every 1% move up (Every Strike) this will induce more buying. For every 1% move down this will induce more selling. Watch VEX chart closely.',
            style={'padding': '20px'})
    ]),
    html.Details([
        html.Summary('State 3: Negative Gamma/Positive Vanna'),
        html.Div(
            'For every 1% move up expect more sellers. Negative Gamma will target the Largest strike so your best opportunities are to short the closest positive gamma strike. Additionally you can identify potential VEX levels to short if they are large.',
            style={'padding': '20px'})
    ]),
    html.Details([
        html.Summary('State 4: Negative Gamma/Negative Vanna'),
        html.Div(
            'For every 1% move down more people will sell. This is short bounces typically less than 50% until the LARGEST gamma strike is hit. Look for largest VEX levels to short or long. Edge play bounce will probably yield a good result.',
            style={'padding': '20px'})
    ]),
    html.Details([
        html.Summary('Vega Total'),
        html.Div(
            'Long Vega - Expecting an increase in implied volatility. Short Vega - Expecting a decrease in implied volatility. Watch the VEX chart and find the correlating levels.',
            style={'padding': '20px'})
    ]),
], style={'fontFamily': 'Arial, sans-serif'})

def print_rate_limits(response):
    used = response.headers.get('X-Ratelimit-Used')
    remaining = response.headers.get('X-Ratelimit-Available')
    limit = response.headers.get('X-Ratelimit-Allowed')
    if used and remaining and limit:
        used = int(used)
        remaining = int(remaining)
        limit = int(limit)
        percentage_used = (used / limit) * 100

        print(f"Rate limits: Used: {used}/{limit} ({percentage_used:.2f}%). Remaining: {remaining}")
    else:
        print("Rate limits information not available")

def get_tradier_credentials(is_sandbox):

    access_token = 'bsYMippYecpJmcK4BYCHaAmaGto0'
    base_url = 'https://sandbox.tradier.com/v1/' if is_sandbox else 'https://api.tradier.com/v1/'

    return access_token, base_url

def get_quote(symbols, is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    symbols_str = ','.join(symbols)
    url = f"{base_url}markets/quotes?symbols={symbols_str}"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    #print_rate_limits(response)
    if response.status_code == 200 and 'quotes' in response.json():
        quotes = response.json()['quotes']
        if 'quote' in quotes:
            return quotes['quote']
    return None


def get_options_expirations(symbol, is_sandbox=False, max_days=1):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    url = f"{base_url}markets/options/expirations?symbol={symbol}"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    #print_rate_limits(response)
    if response.status_code == 200:
        expiration_dates = response.json().get('expirations', {}).get('date', [])
        filtered_dates = []

        pst = pytz.timezone('America/Los_Angeles')
        now_pst = datetime.now(pst)
        cutoff_time = now_pst.replace(hour=14, minute=0, second=0, microsecond=0)

        # Adjust the logic based on the current time and cutoff time
        for date in expiration_dates:
            expiration_date = datetime.strptime(date, '%Y-%m-%d')
            expiration_date_pst = pst.localize(expiration_date)

            # Determine if we should start considering from today or tomorrow
            consider_from_date = now_pst.date() if now_pst <= cutoff_time else now_pst.date() + timedelta(days=1)

            if expiration_date_pst.date() >= consider_from_date and (
                    expiration_date_pst.date() - now_pst.date()).days <= max_days:
                filtered_dates.append(date)

        if filtered_dates:
            return [filtered_dates[0]]
        elif expiration_dates:
            # If no suitable expiration dates are found, return the first available date as a fallback
            return [expiration_dates[0]]
        else:
            # Return an empty list if no expiration dates are available
            return []
    else:
        print(f"Failed to retrieve options expirations for {symbol}. Status code: {response.status_code}")
        return None


def get_options_data(symbol, expiration_date, is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    url = f"{base_url}markets/options/chains?symbol={symbol}&expiration={expiration_date}&greeks=true"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    #print_rate_limits(response)
    if response.status_code == 200:
        options_data = response.json()

        if options_data['options'] is None:
            print(f"No options data found for {symbol}, expiration {expiration_date}.")
            return None

        # Convert options data to DataFrame
        options_df = pd.DataFrame(options_data['options']['option'])

        # Fetch the current price of the symbol to determine the strike range
        current_price = get_qqq_close(is_sandbox)

        # Define the strike range: 15 strikes below and 15 strikes above the current price
        lower_bound = current_price - 25
        upper_bound = current_price + 25

        # Filter the DataFrame for options within the desired strike range
        options_df = options_df[(options_df['strike'] >= lower_bound) & (options_df['strike'] <= upper_bound)]

        return options_df
    else:
        print(f"Failed to retrieve options data for {symbol}, expiration {expiration_date}. Status code: {response.status_code}")
        return None


def get_qqq_close(is_sandbox=False):
    access_token, base_url = get_tradier_credentials(is_sandbox)
    url = f"{base_url}markets/quotes?symbols=QQQ"
    headers = {'Authorization': f'Bearer {access_token}', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        quote_data = response.json().get('quotes', {}).get('quote', {})

        # Convert current UTC time to PST
        current_time_utc = datetime.now(pytz.utc)
        pst_timezone = pytz.timezone('America/Los_Angeles')
        current_time_pst = current_time_utc.astimezone(pst_timezone)

        # Define time ranges in PST
        start_time_use_close = pst_timezone.localize(datetime(current_time_pst.year, current_time_pst.month, current_time_pst.day, 15, 0))
        end_time_use_close = pst_timezone.localize(datetime(current_time_pst.year, current_time_pst.month, current_time_pst.day, 23, 59))
        start_time_use_prevclose = pst_timezone.localize(datetime(current_time_pst.year, current_time_pst.month, current_time_pst.day, 0, 0))
        end_time_use_prevclose = pst_timezone.localize(datetime(current_time_pst.year, current_time_pst.month, current_time_pst.day, 13, 0))

        # Determine which price to use based on current time
        if start_time_use_close <= current_time_pst <= end_time_use_close:
            # Use close_price
            price_to_use = quote_data.get('close')
            print("Using close price")
        elif start_time_use_prevclose <= current_time_pst <= end_time_use_prevclose:
            # Use prevclose
            price_to_use = quote_data.get('prevclose')
            print("Using prevclose price")
        else:
            # Default to close_price if current time doesn't match any condition
            price_to_use = quote_data.get('close')
            print("Using default close price")

        #price_to_use = quote_data.get('close')

        return price_to_use
    else:
        print(f"Failed to retrieve QQQ quote data. Status code: {response.status_code}")
        return None


def calculate_iv(S, K, t, r, price, option_type):
    if option_type == 'call':
        flag = 'c'
    else:
        flag = 'p'

    # implied_volatility = vectorized_implied_volatility(
    #     price, S, K, t, r, [flag], q=0, model='black_scholes_merton', return_as='numpy'
    # )

    implied_volatility = vectorized_implied_volatility(
        price, S, K, t, r, [flag], q=0, model='black', return_as='numpy'
    )

    return implied_volatility[0]


def calculate_t(expiration_date):
    # Set the Eastern Time zone
    eastern_tz = pytz.timezone('US/Eastern')

    # Get the current time in Eastern Time
    current_datetime = datetime.now(eastern_tz)

    # Convert the expiration date to Eastern Time
    expiration_datetime = datetime.strptime(expiration_date, '%Y-%m-%d').replace(tzinfo=eastern_tz)

    # Set the market closing time for the expiration date
    market_close_time = dt_time(20,
                                0)  # Assuming market closes at 4:00 PM ET (adjust as per your market's closing time)
    market_close_datetime = datetime.combine(expiration_datetime.date(), market_close_time).astimezone(eastern_tz)
    time_until_close = float((market_close_datetime - current_datetime).total_seconds()) // 60.0
    # Calculate t based on the time until market close and the number of minutes in a year
    minutes_in_year = 365.25 * 24.00 * 60.00
    t = time_until_close / minutes_in_year if time_until_close is not None else None

    return t


def calculate_vanna(S, K, r, sigma, T):
    #d1 = (math.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
    d1 = (math.log(S / K) + (r - 0 + 0.5 * (sigma ** 2)) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    q = 0

    #vanna = (norm.pdf(d1) * d2) / sigma

    vanna = math.exp(-q * T) * norm.pdf(d1) * (d2 / sigma)

    return vanna


def calculate_greeks(options_df, S, r):
    # S: The price of the underlying asset
    # r: The Interest Free Rate

    # If options_df is empty or None, return immediately
    if options_df is None or options_df.empty:
        return None

    # for every row of the data frame, calculate the delta, gamma, theta, vega nad rho greek floating point values.
    for index, row in options_df.iterrows():
        # Print every column of the row's value first with it's corresponding column name
        # column_names = row.index
        # print(column_names)
        # for column in column_names:
        #     print('{}: {}'.format(column, row[column]))

        # K: The strike price
        K = row['strike']

        # t: The annualized time to expiration.
        t = calculate_t(row['expiration_date'])

        # iv: The Implied Volatility
        #iv = calculate_iv(S, K, t, r, row['ask'], row['option_type'])
        iv = calculate_iv(S, K, t, r, row['last'], row['option_type'])
        #print('iv: {}'.format(iv))

        # q: Annualized Continuous Dividend Yield
        q = 0

        if row['option_type'] == 'call':
            flag = 'c'
        else:
            flag = 'p'

        #greeks = get_all_greeks(flag, S, K, t, r, iv, q, model='black_scholes_merton', return_as='dict')
        greeks = get_all_greeks(flag, S, K, t, r, iv, q, model='black', return_as='dict')
        #print(str(greeks))

        options_df.at[index, 't'] = t
        options_df.at[index, 'implied_volatility'] = iv
        options_df.at[index, 'delta'] = greeks['delta'][0]
        options_df.at[index, 'gamma'] = greeks['gamma'][0]
        options_df.at[index, 'theta'] = greeks['theta'][0]
        options_df.at[index, 'vega'] = greeks['vega'][0]
        options_df.at[index, 'rho'] = greeks['rho'][0]

        vanna = calculate_vanna(S, row['strike'], r, options_df.at[index, 'implied_volatility'], t)
        options_df.at[index, 'vanna'] = vanna

    return options_df


def calculate_adjusted_gex(gamma, oi, volume, max_oi, max_volume, vega, beta, option_type, qqq_close):
    # Normalize volume and OI
    normalized_volume = volume / max_volume if max_volume > 0 else 0
    normalized_oi = oi / max_oi if max_oi > 0 else 0
    combined_metric = 0.3 * normalized_volume + 0.7 * normalized_oi

    # Standard GEX calculation
    gex = gamma * oi * combined_metric * 100 * qqq_close * qqq_close * 0.01

    # Adjusted GEX with Vega
    adjusted_gex = gex * (1 + beta * vega)

    # Handle Puts by making GEX negative
    if option_type == 'put':
        adjusted_gex = -adjusted_gex

    return adjusted_gex

def calculate_shares_for_hedging(total_delta, strategy='neutral'):
    """
    Calculate the number of shares needed to hedge portfolio based on total delta and strategy.

    :param total_delta: Aggregated delta of the portfolio.
    :param strategy: Hedging strategy - 'bullish', 'bearish', or 'neutral'.
    :return: Number of shares to buy/sell. Positive for buying, negative for selling.
    """
    # Neutral strategy aims for delta-neutral hedging
    if strategy == 'neutral':
        return -total_delta  # Sell if positive, buy if negative

    # Adjust the amount of hedging based on strategy
    adjustment_factor = {
        'bullish': 0.9,  # Slightly less hedging if bullish
        'bearish': 1.1,  # Additional hedging if bearish
        'neutral': 1.0,  # Default case, already handled
    }.get(strategy, 1.0)

    # Calculate adjusted shares to hedge
    adjusted_shares = -total_delta * adjustment_factor

    return adjusted_shares


def round_to_nearest_quarter(value):
    return round(value * 4) / 4


@app.callback(
    [Output('output-container', 'children'),
     Output('gauge-chart', 'figure'),
     Output('gex-histogram', 'figure'),
     Output('vex-histogram', 'figure'),
     Output('dex-histogram', 'figure')],
    [Input('interval-component', 'n_intervals')])
def update_output(n_intervals):
    # Placeholder for actual data fetching and calculation based on qqq_close
    # For demonstration, using mock data
    # Fetch options data
    symbol = "QQQ"
    symbol_data = get_quote([symbol])
    #qqq_close = get_qqq_close()
    qqq_close = 502.96
    nq_close = 20917
    expiration_dates = get_options_expirations(symbol)
    options_df = get_options_data(symbol, expiration_dates[0])
    ratio = round(nq_close / qqq_close, 2)
    beta = 0.1


    if options_df is not None and symbol_data is not None:
        # Calculate Greeks and add to DataFrame
        dataframe = calculate_greeks(options_df, qqq_close, .0548)

        # # Apply adjusted GEX calculation for each row
        # max_oi = dataframe['open_interest'].max()
        # max_volume = dataframe['volume'].max()
        #
        # # Calculate GEX for Calls
        # dataframe['GEX_Calls'] = dataframe.apply(
        #     lambda row: calculate_adjusted_gex(
        #         row['gamma'], row['open_interest'], row['volume'], max_oi, max_volume, row['vega'], beta,
        #         row['option_type'], qqq_close
        #     ) if row['option_type'] == 'call' else 0, axis=1
        # )
        #
        # # Calculate GEX for Puts
        # dataframe['GEX_Puts'] = dataframe.apply(
        #     lambda row: calculate_adjusted_gex(
        #         row['gamma'], row['open_interest'], row['volume'], max_oi, max_volume, row['vega'], beta,
        #         row['option_type'], qqq_close
        #     ) if row['option_type'] == 'put' else 0, axis=1
        # )

        # Calculate GEX for Calls
        dataframe['GEX_Calls'] = options_df.apply(
            lambda row: row['gamma'] * row['open_interest'] * 100 * qqq_close * qqq_close * 0.01 if row['option_type'] == 'call' else 0, axis=1)

        # Calculate GEX for Puts
        dataframe['GEX_Puts'] = options_df.apply(
            lambda row: row['gamma'] * row['open_interest'] * 100 * qqq_close * qqq_close * 0.01 * -1 if row['option_type'] == 'put' else 0, axis=1)

        # Calculate VEX for calls
        dataframe['VEX_Calls'] = options_df.apply(
            lambda row: row['vanna'] * row['open_interest'] * row['implied_volatility'] * qqq_close if row['option_type'] == 'call' else 0, axis=1)

        # Calculate VEX for puts
        dataframe['VEX_Puts'] = options_df.apply(
            lambda row: row['vanna'] * row['open_interest'] * row['implied_volatility'] * qqq_close if row['option_type'] == 'put' else 0, axis=1)

        # Calculate DEX for Calls
        dataframe['DEX_Calls'] = options_df.apply(
            lambda row: row['delta'] * row['open_interest'] * 100 if row['option_type'] == 'call' else 0, axis=1)

        # Calculate DEX for Puts
        dataframe['DEX_Puts'] = options_df.apply(
            lambda row: row['delta'] * row['open_interest'] * 100 if row['option_type'] == 'put' else 0, axis=1)

        # Calculate Vega for Calls
        dataframe['Vega_Calls'] = options_df.apply(
            lambda row: row['vega'] * row['open_interest'] * 100 if row['option_type'] == 'call' else 0, axis=1)

        # Calculate Vega for Puts
        dataframe['Vega_Puts'] = options_df.apply(
            lambda row: row['vega'] * row['open_interest'] * -100 if row['option_type'] == 'put' else 0, axis=1)

        # Raw aggregate Delta per strike
        dataframe['Delta_Calls'] = options_df.apply(
            lambda row: row['delta'] * row['open_interest'] / (1*100) if row['option_type'] == 'call' else 0, axis=1)

        dataframe['Delta_Puts'] = options_df.apply(
            lambda row: row['delta'] * row['open_interest'] / (1*100) if row['option_type'] == 'put' else 0, axis=1)

        dataframe['aggDelta'] = dataframe['Delta_Calls'] + dataframe['Delta_Puts']
        total_delta = dataframe['aggDelta'].sum()

        shares_to_hedge = calculate_shares_for_hedging(total_delta, 'neutral')
        #(f"Shares to hedge: {shares_to_hedge}")


        # Calculate the totals
        dataframe['VEX'] = dataframe['VEX_Calls'] + dataframe['VEX_Puts']
        dataframe['GEX'] = dataframe['GEX_Calls'] + dataframe['GEX_Puts']
        dataframe['DEX'] = dataframe['DEX_Calls'] + dataframe['DEX_Puts']
        dataframe['Vega'] = dataframe['Vega_Calls'] + dataframe['Vega_Puts']

        # GROUPING GEX
        gex_by_strike = dataframe.groupby('strike')['GEX'].sum().reset_index()
        gex_by_strike['adjusted_strike'] = gex_by_strike['strike'] * ratio
        gex_by_strike['optimized_strike'] = gex_by_strike['adjusted_strike'].apply(round_to_nearest_quarter)
        total_gex = round(gex_by_strike['GEX'].sum(), 2)
        total_gex_formatted = format(total_gex, ",.2f")

        # GROUPING VEX
        vex_by_strike = dataframe.groupby('strike')['VEX'].sum().reset_index()
        vex_by_strike['adjusted_strike'] = vex_by_strike['strike'] * ratio
        vex_by_strike['optimized_strike'] = vex_by_strike['adjusted_strike'].apply(round_to_nearest_quarter)
        total_vex = round(vex_by_strike['VEX'].sum(), 2)
        total_vex_formatted = format(total_vex, ",.2f")

        # GROUPING DEX
        dex_by_strike = dataframe.groupby('strike')['DEX'].sum().reset_index()
        dex_by_strike['adjusted_strike'] = dex_by_strike['strike'] * ratio
        dex_by_strike['optimized_strike'] = dex_by_strike['adjusted_strike'].apply(round_to_nearest_quarter)
        total_dex = round(dex_by_strike['DEX'].sum(), 2)
        total_dex_formatted = format(total_dex, ",.2f")
        dex_by_strike['DEX_Calls'] = dex_by_strike['DEX'].apply(lambda x: x if x > 0 else 0)
        dex_by_strike['DEX_Puts'] = dex_by_strike['DEX'].apply(lambda x: x if x < 0 else 0)
        dex_by_strike['DEX_Total'] = dex_by_strike['DEX_Calls'] + dex_by_strike['DEX_Puts']
        dex_by_strike['sharesneeded'] = dex_by_strike['DEX_Total'].apply(lambda x: x * shares_to_hedge)

        # GROUPING Vega
        vega_by_strike = dataframe.groupby('strike')['Vega'].sum().reset_index()
        vega_by_strike['adjusted_strike'] = vega_by_strike['strike'] * ratio
        vega_by_strike['optimized_strike'] = vega_by_strike['adjusted_strike'].apply(round_to_nearest_quarter)
        total_vega = round(vega_by_strike['Vega'].sum(), 2)
        total_vega_formatted = format(total_vega, ",.2f")
        vega_by_strike['Vega_Calls'] = vega_by_strike['Vega'].apply(lambda x: x if x > 0 else 0)
        vega_by_strike['Vega_Puts'] = vega_by_strike['Vega'].apply(lambda x: x if x < 0 else 0)

        # Calculating total Bullish Percentage Gauge for DEX
        total_dex_calls = options_df['DEX_Calls'].sum()
        total_dex_puts = options_df['DEX_Puts'].sum()
        total_pct_Call_dex = abs(total_dex_calls)
        total_pct_Put_dex = abs(total_dex_puts)
        total_absolute_dex = total_pct_Call_dex + total_pct_Put_dex
        bullish_pct_dex = (total_pct_Call_dex / total_absolute_dex * 100) if total_absolute_dex != 0 else 0

        # Color Columns
        gex_by_strike['color'] = gex_by_strike['GEX'].apply(lambda x: 'green' if x > 0 else 'red')
        vex_by_strike['color'] = vex_by_strike['VEX'].apply(lambda x: 'green' if x > 0 else 'red')
        dex_by_strike['color'] = dex_by_strike['DEX'].apply(lambda x: 'green' if x > 0 else 'red')
        vega_by_strike['color'] = vega_by_strike['Vega'].apply(lambda x: 'green' if x > 0 else 'red')

        # Convert 'optimized_strike' to a numeric type (float)
        gex_by_strike['optimized_strike'] = pd.to_numeric(gex_by_strike['optimized_strike'], errors='coerce')
        vex_by_strike['optimized_strike'] = pd.to_numeric(vex_by_strike['optimized_strike'], errors='coerce')
        dex_by_strike['optimized_strike'] = pd.to_numeric(dex_by_strike['optimized_strike'], errors='coerce')
        vega_by_strike['optimized_strike'] = pd.to_numeric(vega_by_strike['optimized_strike'], errors='coerce')

        #filter dataframe for strikes 10 strikes above and below the qqq_close
        dataframe_adjusted = gex_by_strike.sort_values(by='optimized_strike', ascending=True)
        vex_dataframe_adjusted = vex_by_strike.sort_values(by='optimized_strike', ascending=True)
        dex_dataframe_adjusted = dex_by_strike.sort_values(by='optimized_strike', ascending=True)
        vega_dataframe_adjusted = vega_by_strike.sort_values(by='optimized_strike', ascending=True)

        gex_skewness = dataframe_adjusted['GEX'].skew()
        vex_skewness = vex_dataframe_adjusted['VEX'].skew()


        # Find the nearest strike just below nq_close - 1000
        left_strikes = gex_by_strike[gex_by_strike['optimized_strike'] <= nq_close - 1000]
        if not left_strikes.empty:
            left_strike = left_strikes.iloc[-1]['optimized_strike']  # Last one before nq_close - 1000
        else:
            left_strike = gex_by_strike.iloc[0]['optimized_strike']  # Default to the first strike if none found

        # Find the nearest strike just above nq_close + 1000
        right_strikes = gex_by_strike[gex_by_strike['optimized_strike'] >= nq_close + 1000]
        if not right_strikes.empty:
            right_strike = right_strikes.iloc[0]['optimized_strike']  # First one after nq_close + 1000
        else:
            right_strike = gex_by_strike.iloc[-1]['optimized_strike']  # Default to the last strike if none found


        #---------------------------------------------------------------------------------------------------------------#

        fig = px.bar(dataframe_adjusted, x='optimized_strike', y='GEX',
                     title="GEX at Each Strike Converted to NQ Prices",
                     color='color', color_discrete_map={'green': 'green', 'red': 'red'})

        # Add skewness annotation to the GEX bar plot
        fig.add_annotation(
            x=0.95, y=0.95,  # Positioning the annotation at the top right corner of the plot area
            text=f"Skewness: {gex_skewness:.2f}",
            showarrow=False,
            xref="paper", yref="paper",  # Using 'paper' refers to the whole figure area
            align="right",
            font=dict(
                size=12,
                color="black"  # Ensure contrast with background
            ),
            bgcolor="white",
            bordercolor="black",
            borderpad=4
        )

        fig.update_layout(template="plotly_dark", xaxis=dict(
            tickmode='array',
            tickvals=dataframe_adjusted['optimized_strike'],
            ticktext=[f"{x:.2f}" for x in dataframe_adjusted['optimized_strike']],
            rangeslider=dict(
                visible=True
            ),
            type='linear',
            range = [left_strike, right_strike]
        ))

        # Generate the histogram vex
        fig2 = px.bar(vex_dataframe_adjusted, x='optimized_strike', y='VEX',
                     title="VEX at Each Strike Converted to NQ Prices",
                     color='color', color_discrete_map={'green': 'green', 'red': 'red'})

        # Add skewness annotation to the VEX bar plot
        fig2.add_annotation(
            x=0.95, y=0.95,  # Positioning the annotation at the top right corner of the plot area
            text=f"Skewness: {vex_skewness:.2f}",
            showarrow=False,
            xref="paper", yref="paper",  # Using 'paper' refers to the whole figure area
            align="right",
            font=dict(
                size=12,
                color="black"  # Ensure contrast with background
            ),
            bgcolor="white",
            bordercolor="black",
            borderpad=4
        )

        fig2.update_layout(template="plotly_dark", xaxis=dict(
            tickmode='array',
            tickvals=vex_dataframe_adjusted['optimized_strike'],
            ticktext=[f"{x:.2f}" for x in vex_dataframe_adjusted['optimized_strike']],
            rangeslider=dict(
                visible=True
            ),
            type='linear',
            range=[left_strike, right_strike]
        ))

        fig3 = go.Figure()

        # Determine the color of each bar based on DEX_Total value
        bar_colors = ['green' if x > 0 else 'red' for x in dex_by_strike['DEX_Total']]

        # Add DEX_Total trace with dynamic coloring
        fig3.add_trace(go.Bar(x=dex_by_strike['optimized_strike'],
                              y=dex_by_strike['DEX_Total'],  # Plotting DEX_Total
                              name='DEX Total',
                              marker_color=bar_colors))  # Setting bar colors based on DEX_Total value

        # Optionally, add Shares to Hedge as a line on a secondary y-axis
        fig3.add_trace(go.Scatter(x=dex_by_strike['optimized_strike'],
                                  y=dex_by_strike['sharesneeded'],
                                  mode='lines+markers',  # Adding markers for clarity
                                  name='Shares to Hedge',
                                  line=dict(color='white', width=2),
                                  yaxis='y2'))  # Linking to the secondary y-axis

        # Update layout with a secondary y-axis for Shares to Hedge
        fig3.update_layout(
            template="plotly_dark",
            xaxis=dict(
                tickmode='array',
                tickvals=dex_by_strike['optimized_strike'],
                ticktext=[f"{x:.2f}" for x in dex_by_strike['optimized_strike']],
                title="Optimized Strike"
            ),
            yaxis=dict(
                title="DEX Total Value",
            ),
            yaxis2=dict(
                title="Shares to Hedge",
                overlaying='y',
                side='right',
                showgrid=False,  # Optionally hide the gridlines for the secondary y-axis
            )
        )

        # Create the gauge chart using the calculated total_pct_dex
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=bullish_pct_dex,
            number={'suffix': "%", 'font': {'size': 80}},
            title={'text': "Bullish DEX vs Bearish DEX"},
            delta={'reference': 10, 'increasing': {'color': "Green"}, 'decreasing': {'color': "Red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "Green"},
                'bar': {'color': "Green"},  # Color of the needle or indicator bar
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 100], 'color': 'White'},  # Bearish part of the gauge
                ],
                # Additional gradient for aesthetic (optional)
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            domain={'x': [0, 1], 'y': [0, 1]}
        ))

        # Adjusted annotations if totals are not pre-converted to "millions" format
        fig4.add_annotation(x=0.23, y=0.45, text=f"Bullish DEX<br>{total_pct_Call_dex / 1e6:.2f}M", showarrow=False,
                            font={'color': "White"})
        fig4.add_annotation(x=0.77, y=0.45, text=f"Bearish DEX<br>{total_pct_Put_dex / 1e6:.2f}M", showarrow=False,
                            font={'color': "White"})

        fig4.update_layout(
            height=300,
            margin={'t': 10, 'b': 10, 'l': 10, 'r': 10},
            template="plotly_dark"
        )

        output_children = [
            html.P(f"Expiration Date: {expiration_dates[0]}"),
            html.P(f"Close Price: {qqq_close}"),
            html.P(f"Total GEX: {total_gex_formatted}"),
            html.P(f"Total VEX: {total_vex_formatted}"),
            html.P(f"Total DEX: {total_dex_formatted}"),
            html.P(f"Total Vega: {total_vega_formatted}"),
            html.P(f"Conversion Rate: {ratio}")
        ]

        return output_children,fig4, fig, fig2, fig3
    else:
        return 'Enter values and press submit', go.Figure(
            data=[],
            layout=go.Layout(
                title="No data to display. Enter values and press submit.",
                xaxis=dict(title="Strike"),
                yaxis=dict(title="GEX"),
                template="plotly_dark"
            )
        )

if __name__ == '__main__':
    app.run_server(debug=True)
