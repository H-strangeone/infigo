from dotenv import load_dotenv
from supabase import create_client
import os
import bcrypt
from datetime import datetime, timezone

load_dotenv(dotenv_path=".env")

SUPABASE_URL = os.getenv("SUPABASE_URL") 
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_user_by_employee_id(emp_id):
    response = supabase.table("users").select("*").eq("employee_id", emp_id).execute()
    return response.data[0] if response.data else None

def insert_user(emp_id, email, hashed_password):
    return supabase.table("users").insert({
        "employee_id": emp_id,
        "email": email,
        "password": hashed_password,
        "login_attempts": 0,
        "locked": False
    }).execute()

def update_otp(emp_id, otp, expiry):
    return supabase.table("users").update({
        "otp": otp,
        "otp_valid_until": expiry
    }).eq("employee_id", emp_id).execute()

def reset_password(emp_id, otp, new_password):
    user = get_user_by_employee_id(emp_id)
    if not user or user["otp"] != otp or datetime.now(timezone.utc).isoformat() > user["otp_valid_until"]:
        return False

    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    return supabase.table("users").update({
        "password": hashed,
        "otp": None,
        "otp_valid_until": None,
        "login_attempts": 0,
        "locked": False
    }).eq("employee_id", emp_id).execute()

    return True
def increment_login_attempts(emp_id):
    user = get_user_by_employee_id(emp_id)
    attempts = user.get("login_attempts", 0) + 1
    locked = attempts >= 3
    return supabase.table("users").update({
        "login_attempts": attempts,
        "locked": locked
    }).eq("employee_id", emp_id).execute()

def reset_login_attempts(emp_id):
    return supabase.table("users").update({
        "login_attempts": 0,
        "locked": False
    }).eq("employee_id", emp_id).execute()
