import streamlit as st
import pandas as pd
import time
from datetime import datetime

# Get current date and time
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

# Read the CSV file
csv_file_path = "Attendance/Attendance_" + date + ".csv"
d_frame = pd.read_csv(csv_file_path)

# Display the DataFrame with the maximum value in each column highlighted
st.dataframe(d_frame.style.highlight_max(axis=1))