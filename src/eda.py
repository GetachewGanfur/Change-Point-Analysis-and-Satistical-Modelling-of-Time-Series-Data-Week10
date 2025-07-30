import matplotlib.pyplot as plt

class EDA:
    def __init__(self, df):
        self.df = df

    def plot_price(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Date'], self.df['Price'])
        plt.title('Brent Oil Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.show()