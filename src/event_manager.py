import pandas as pd

class EventManager:
    def __init__(self, event_path):
        self.event_path = event_path

    def load_events(self):
        df = pd.read_csv(self.event_path, parse_dates=['Date'], dayfirst=True)
        return df