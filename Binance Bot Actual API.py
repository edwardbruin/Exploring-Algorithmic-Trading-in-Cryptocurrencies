# Importing libraries
from binance.client import Client
import configparser
import pandas as pd

# Loading keys from config file
config = configparser.ConfigParser()
config.read_file(open('secret_keys.config'))
actual_api_key = config.get('BINANCE', 'ACTUAL_API_KEY')
actual_secret_key = config.get('BINANCE', 'ACTUAL_SECRET_KEY')

client = Client(actual_api_key, actual_secret_key)

# Getting earliest timestamp availble (on Binance)
earliest_timestamp = client._get_earliest_valid_timestamp('ETHUSDT', '1d')  # Here "ETHUSDT" is a trading pair and "1d" is time interval
print(earliest_timestamp)

# Getting historical data (candle data or kline)
candle = client.get_historical_klines("ETHUSDT", "1d", earliest_timestamp, limit=1000)

print(candle[1])

eth_df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
eth_df.dateTime = pd.to_datetime(eth_df.dateTime, unit='ms')
eth_df.closeTime = pd.to_datetime(eth_df.closeTime, unit='ms')
eth_df.set_index('dateTime', inplace=True)
eth_df.to_csv('eth_candle.csv')
eth_df.to_csv(r'C:\Users\devanshipatel\Desktop\export_eth_candle.csv', index = False, header=True)

print(eth_df.tail(10))

# Importing libraries
from binance.client import Client
import configparser
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor

# Loading keys from config file
config = configparser.ConfigParser()
config.read_file(open('secret_keys.config'))
actual_api_key = config.get('BINANCE', 'ACTUAL_API_KEY')
actual_secret_key = config.get('BINANCE', 'ACTUAL_SECRET_KEY')

client = Client(actual_api_key, actual_secret_key)


def streaming_data_process(msg):

    print(f"message type: {msg['e']}")
    print(f"close price: {msg['c']}")
    print(f"best ask price: {msg['a']}")
    print(f"best bid price: {msg['b']}")
    print("---------------------------")


# Starting the WebSocket
bm = BinanceSocketManager(client)
conn_key = bm.start_symbol_ticker_socket('ETHUSDT', streaming_data_process)
bm.start()


# Stopping websocket
bm.stop_socket(conn_key)

# Websockets utilise a reactor loop from the Twisted library. Using the stop method above
# will stop the websocket connection but it won’t stop the reactor loop so your code may
# not exit when you expect.

# When you need to exit
reactor.stop()


#if __name__ == "__main__":