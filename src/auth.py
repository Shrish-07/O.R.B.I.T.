import sqlite3
from pathlib import Path
import bcrypt
from datetime import datetime
import json

DB_PATH = Path('data') / 'users.db'
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password_hash TEXT,
        created_utc TEXT,
        data JSON
    )''')
    c.commit()
    return c

def create_user(email: str, password: str):
    conn = _conn()
    cur = conn.cursor()
    pw_hash = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt()).decode('utf8')
    now = datetime.utcnow().isoformat() + 'Z'
    try:
        cur.execute('INSERT INTO users (email, password_hash, created_utc, data) VALUES (?,?,?,?)', (email, pw_hash, now, json.dumps({})))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(email: str, password: str):
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT id, password_hash FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    if not row:
        return None
    uid, pw_hash = row
    try:
        ok = bcrypt.checkpw(password.encode('utf8'), pw_hash.encode('utf8'))
    except Exception:
        ok = False
    return uid if ok else None

def get_user_data(user_id: int):
    conn = _conn()
    cur = conn.cursor()
    cur.execute('SELECT data FROM users WHERE id = ?', (user_id,))
    row = cur.fetchone()
    if not row:
        return None
    return json.loads(row[0])

def save_user_data(user_id: int, data: dict):
    conn = _conn()
    cur = conn.cursor()
    cur.execute('UPDATE users SET data = ? WHERE id = ?', (json.dumps(data), user_id))
    conn.commit()
    return True
