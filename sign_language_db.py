import psycopg2
from datetime import datetime, timezone
def get_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="sign_language_db",
            user="postgres",
            password="4004",
            port="5432"
        )
        print("✅ Connected to database")
        return conn
    except Exception as e:
        print("❌ Database connection failed:", e)
        return None

def insert_prediction(label, accuracy):
    conn = get_connection()  # now works
    if not conn:
        print("⚠ Skipping insert - no DB connection")
        return

    try:
        cur = conn.cursor()
        # now_utc = datetime.now(timezone.utc)
        # date_utc = now_utc.date()
        # time_utc = now_utc.time()
        cur.execute("""
            INSERT INTO public.sign_language_predictions (prediction_date, prediction_time, label, accuracy)
            VALUES (CURRENT_DATE, CURRENT_TIME, %s, %s)
        """, (label, float(accuracy)))  # convert numpy.float32 to float
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Saved: {label} ({accuracy*100:.1f}%)")
    except Exception as e:
        print("❌ Insert failed:", e)