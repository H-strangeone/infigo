from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import os
import bcrypt
from datetime import datetime, timedelta, timezone
import random 
from db import (
    get_user_by_employee_id,
    insert_user,
    update_otp,
    reset_password,
    increment_login_attempts,
    reset_login_attempts,
    supabase 
)
load_dotenv(dotenv_path=".env")


SENDER_EMAIL = os.getenv("GMAIL_SENDER")  # your@gmail.com
SENDER_PASS = os.getenv("GMAIL_APP_PASSWORD")  # Gmail App Password

def send_otp_email(to_email, otp):
    msg = MIMEText(f"Your OTP for login is: {otp}")
    msg["Subject"] = "Your OTP Code"
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    print(f"Trying to send email to {to_email} using {SENDER_EMAIL}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASS)
            server.send_message(msg)
        print("✅ OTP email sent.")
        return True
    except Exception as e:
        print(f"❌ Failed to send OTP: {e}")
        return False


def generate_otp():
    return str(random.randint(100000, 999999))

# def signup(emp_id, password, email):
#     existing = get_user_by_employee_id(emp_id)
#     if existing:
#         return False
#     hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

#     # Generate OTP and try to send before inserting
#     otp = generate_otp()
#     expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

#     #  Only insert if email sends successfully
#     if send_otp_email(email, otp):
#         insert_user(emp_id, email, hashed)
#         update_otp(emp_id, otp, expiry)
#         return True
#     else:
#         print("Email failed. Not inserting user.")
#         return False
def signup(emp_id, password, email):
    existing = get_user_by_employee_id(emp_id)
    if existing:
        return False

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    insert_user(emp_id, email, hashed)
    otp = generate_otp()
    expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    update_otp(emp_id, otp, expiry)

    if send_otp_email(email, otp):
        return True
    else:
        supabase.table("users").delete().eq("employee_id", emp_id).execute()
        return False

def login(emp_id, password):
    user = get_user_by_employee_id(emp_id)
    if not user:
        return False

    if user.get("login_attempts", 0) >= 3:
        print("🚫 Too many failed attempts. Account locked.")
        return False

    if bcrypt.checkpw(password.encode(), user["password"].encode()):
        reset_login_attempts(emp_id)  
        return True
    else:
        increment_login_attempts(emp_id)  
        return False


from datetime import datetime, timezone

def verify_otp(emp_id, otp):
    user = get_user_by_employee_id(emp_id)
    if not user:
        return False

    if user['otp'] == otp and datetime.now(timezone.utc).isoformat() < user['otp_valid_until']:
        return True
    else:
        
        supabase.table("users").delete().eq("employee_id", emp_id).execute()
        return False

def send_reset_otp(emp_id):
    user = get_user_by_employee_id(emp_id)
    if not user:
        return False

    otp = generate_otp()
    expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    update_otp(emp_id, otp, expiry)

    return send_otp_email(user["email"], otp)




