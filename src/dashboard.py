import streamlit as st
import pandas as pd

class Dashboard:
    def __init__(self, df, events):
        self.df = df
        self.events = events

    def run(self):
        st.title("Brent Oil Price Change Point Analysis")
        st.line_chart(self.df.set_index('Date')['Price'])
        st.write("Key Events")
        st.dataframe(self.events)