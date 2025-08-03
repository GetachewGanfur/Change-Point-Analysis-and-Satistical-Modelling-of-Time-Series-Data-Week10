import pandas as pd
import numpy as np
from datetime import datetime

class EventManager:
    def __init__(self):
        """Initialize with predefined major events affecting oil prices"""
        self.events = self._create_event_database()
    
    def _create_event_database(self):
        """Create a comprehensive database of major events affecting oil prices"""
        events_data = [
            # Gulf War (1990-1991)
            {'date': '1990-08-02', 'event': 'Iraq invades Kuwait', 'category': 'Conflict', 'impact': 'High'},
            {'date': '1991-01-17', 'event': 'Operation Desert Storm begins', 'category': 'Conflict', 'impact': 'High'},
            
            # Asian Financial Crisis (1997-1998)
            {'date': '1997-07-02', 'event': 'Asian Financial Crisis begins', 'category': 'Economic', 'impact': 'Medium'},
            
            # 9/11 and aftermath (2001)
            {'date': '2001-09-11', 'event': '9/11 terrorist attacks', 'category': 'Geopolitical', 'impact': 'High'},
            
            # Iraq War (2003)
            {'date': '2003-03-20', 'event': 'Iraq War begins', 'category': 'Conflict', 'impact': 'High'},
            
            # Global Financial Crisis (2008)
            {'date': '2008-09-15', 'event': 'Lehman Brothers bankruptcy', 'category': 'Economic', 'impact': 'High'},
            
            # Arab Spring (2011)
            {'date': '2011-01-25', 'event': 'Arab Spring begins', 'category': 'Geopolitical', 'impact': 'Medium'},
            
            # Libyan Civil War (2011)
            {'date': '2011-02-17', 'event': 'Libyan Civil War begins', 'category': 'Conflict', 'impact': 'Medium'},
            
            # OPEC Production Cuts (2016)
            {'date': '2016-11-30', 'event': 'OPEC agrees to production cuts', 'category': 'OPEC', 'impact': 'High'},
            
            # US-China Trade War (2018)
            {'date': '2018-07-06', 'event': 'US-China trade war begins', 'category': 'Economic', 'impact': 'Medium'},
            
            # COVID-19 Pandemic (2020)
            {'date': '2020-03-11', 'event': 'COVID-19 declared pandemic', 'category': 'Economic', 'impact': 'High'},
            
            # Russia-Ukraine War (2022)
            {'date': '2022-02-24', 'event': 'Russia invades Ukraine', 'category': 'Conflict', 'impact': 'High'},
            
            # OPEC+ Production Cuts (2022)
            {'date': '2022-10-05', 'event': 'OPEC+ announces major production cuts', 'category': 'OPEC', 'impact': 'High'},
            
            # Additional significant events
            {'date': '1998-12-17', 'event': 'OPEC production cuts', 'category': 'OPEC', 'impact': 'Medium'},
            {'date': '2005-08-29', 'event': 'Hurricane Katrina', 'category': 'Natural Disaster', 'impact': 'Medium'},
            {'date': '2014-06-20', 'event': 'ISIS captures Mosul', 'category': 'Conflict', 'impact': 'Medium'},
            {'date': '2015-07-14', 'event': 'Iran nuclear deal', 'category': 'Geopolitical', 'impact': 'Medium'},
            {'date': '2018-05-08', 'event': 'US withdraws from Iran nuclear deal', 'category': 'Geopolitical', 'impact': 'Medium'},
            {'date': '2019-09-14', 'event': 'Saudi oil facilities attacked', 'category': 'Conflict', 'impact': 'High'},
            {'date': '2020-04-20', 'event': 'Oil price goes negative', 'category': 'Economic', 'impact': 'High'},
            {'date': '2021-11-23', 'event': 'US releases strategic petroleum reserve', 'category': 'Geopolitical', 'impact': 'Medium'}
        ]
        
        df = pd.DataFrame(events_data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_events_in_period(self, start_date, end_date):
        """Get events within a specific date range"""
        mask = (self.events['date'] >= start_date) & (self.events['date'] <= end_date)
        return self.events[mask]
    
    def get_events_by_category(self, category):
        """Get events by category (Conflict, Economic, OPEC, etc.)"""
        return self.events[self.events['category'] == category]
    
    def get_events_by_impact(self, impact_level):
        """Get events by impact level (High, Medium, Low)"""
        return self.events[self.events['impact'] == impact_level]
    
    def find_nearest_event(self, target_date, days_threshold=30):
        """Find the nearest event to a given date within a threshold"""
        target_date = pd.to_datetime(target_date)
        self.events['days_diff'] = abs((self.events['date'] - target_date).dt.days)
        nearest = self.events[self.events['days_diff'] <= days_threshold]
        return nearest.sort_values('days_diff')
    
    def get_all_events(self):
        """Get all events"""
        return self.events.copy()