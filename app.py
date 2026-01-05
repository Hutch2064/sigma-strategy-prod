import os
import sqlite3
import streamlit as st
import bcrypt

# ============================================================
# SQLITE SETUP (PERSISTENT ON FLY.IO)
# ============================================================

DB_DIR = "/data"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "users.db")

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
        """)

init_db()

# ============================================================
# SIMPLE TEST UI
# ============================================================

st.title("Fly.io + SQLite Persistence Test")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Create User"):
    if not username or not password:
        st.error("Both fields required")
    else:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, name, password_hash) VALUES (?, ?, ?)",
                    (username, username, hashed),
                )
            st.success("User created and persisted.")
        except sqlite3.IntegrityError:
            st.error("User already exists.")

if st.button("List Users"):
    with get_db() as conn:
        users = conn.execute("SELECT username FROM users").fetchall()
    st.write(users)
