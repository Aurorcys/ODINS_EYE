from realistic_backtester.Data_loader import DataLoader
import pandas as pd

dl = DataLoader()
data = dl.load_hourly_data("AAPL", days_back=10)

print(f"\nFirst 3 rows:")
print(data[['open', 'high', 'low', 'close', 'volume', 'bid', 'ask']].head(3))
print(f'\nCheck Jan 1, 2024 (holiday):')
print(data[data.index.date == pd.Timestamp('2024-01-01').date()])


#to run this boy: python3 -m realistic_backtester.test