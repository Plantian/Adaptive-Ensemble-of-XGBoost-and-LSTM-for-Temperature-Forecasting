# data visualiaztions

import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

def load_and_process_data(filename):
    df = pd.read_csv(filename)
    aal_df = df[(df['Name'] == 'GILD') & (df['date'] >= '2013-01-01') & (df['date'] <= '2018-12-31')]
    aal_df['date'] = pd.to_datetime(aal_df['date'])
    aal_df = aal_df.sort_values('date')
    aal_df = aal_df.reset_index(drop=True)
    aal_df['date_num'] = mdates.date2num(aal_df['date'])
    aal_df['MA30'] = aal_df['close'].rolling(window=30).mean()
    return aal_df

def create_stock_chart(data):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    ohlc = data[['date_num', 'open', 'high', 'low', 'close']].values
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup='g', colordown='r', alpha=0.8)
    ax1.plot(data['date_num'], data['MA30'], 'b-', linewidth=1.5, label='30-Day MA')
    ax1.set_title('GILD Stock Price (2013-2018)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    ax2.bar(data['date_num'], data['volume'], width=0.6, color=[('g' if close > open else 'r') for close, open in zip(data['close'], data['open'])])
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('GILD_stock_visualization.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    filename = 'D:\\Vs Code\\all_stocks_5yr.csv'
    stock_data = load_and_process_data(filename)
    if stock_data.empty:
        print("no such files and documents")
    else:
        create_stock_chart(stock_data)
        print("visualiaztions has already done: saving as Path: GILD_stock_visualization.png")