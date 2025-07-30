import pandas as pd

class DataLoader:
    def __init__(self, price_path):
        if not isinstance(price_path, str) or not price_path:
            raise ValueError("price_path must be a non-empty string")
        self.price_path = price_path

    def load_data(self):
        df = pd.read_csv(self.price_path, parse_dates=['Date'], dayfirst=False)
        df = df.sort_values('Date').reset_index(drop=True)
        return df