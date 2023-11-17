import yfinance as yf
symbol = "AAPL"
ticker = yf.Ticker(symbol)

data = []

while(
currentPrice = ticker.history().tail(1)["Close"].iloc[0]
