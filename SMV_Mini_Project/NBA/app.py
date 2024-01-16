import sqlite3 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scraping import *

class MainApp:
    def __init__(self):
        # Constructor code here

    def run(self):
        # Main application logic here
        print("Hello, this is the main class!")

class Data:
    def __init__(self):
        # Constructor code here
        if os.path.isfile('SMV_Mini_Project/data.db'):
            initialize = False
        else:
            initialize = True
        self.conn = sqlite3.connect('SMV_Mini_Project/data.db')

    def get_data():
        
        

if __name__ == "__main__":
    app = MainApp()
    app.run()