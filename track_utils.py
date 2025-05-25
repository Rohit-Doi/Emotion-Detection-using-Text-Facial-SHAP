import sqlite3
from datetime import datetime
import pytz

IST = pytz.timezone('Asia/Kolkata')

def create_connection():
    conn = sqlite3.connect('tracker.db')
    return conn

def create_page_visited_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS page_visited (
                        page_name TEXT,
                        time_of_visit TEXT
                    )''')
    conn.commit()
    conn.close()

def add_page_visited_details(page_name, time_of_visit):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO page_visited (page_name, time_of_visit) VALUES (?, ?)",
                   (page_name, time_of_visit.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def view_all_page_visited_details():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM page_visited")
    rows = cursor.fetchall()
    conn.close()
    return rows

def create_emotionclf_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS emotionclf (
                        text TEXT,
                        prediction TEXT,
                        confidence REAL,
                        time_of_prediction TEXT
                    )''')
    conn.commit()
    conn.close()

def add_prediction_details(text, prediction, confidence, time_of_prediction):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO emotionclf (text, prediction, confidence, time_of_prediction) VALUES (?, ?, ?, ?)",
                   (text, prediction, float(confidence), time_of_prediction.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def view_all_prediction_details():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM emotionclf")
    rows = cursor.fetchall()
    conn.close()
    return rows