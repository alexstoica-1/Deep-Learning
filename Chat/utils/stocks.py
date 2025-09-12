import random 

def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']

    print(random.sample(stocks, 3))