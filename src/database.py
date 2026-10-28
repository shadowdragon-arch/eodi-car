"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
Milestone 5, 6 & 8: SQLite Database with Smart Auto-Gate, Live Fee Timer & Validation
"""

import sqlite3
import os
from datetime import datetime
from fee_calculator import calculate_fee
from ocr_reader import smart_malaysian_plate_clean, normalise

DB_PATH = os.path.join("data", "eodicar_sessions.db")

# Auto-confirm without operator intervention when confidence >= 75%
AUTO_CONFIRM_THRESHOLD = 0.75


def _to_pct(conf: float) -> float:
    """Normalizes confidence into a 0.0 - 100.0 percentage float."""
    if conf is None:
        return 0.0
    return round(conf * 100.0, 1) if conf <= 1.0 else round(float(conf), 1)


def init_db():
    """Creates the database file and table if they don't exist, and cleans invalid test rows."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parking_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP,
            status TEXT DEFAULT 'ACTIVE',
            confidence REAL,
            image_path TEXT
        )
    ''')
    cursor.execute("DELETE FROM parking_sessions WHERE LENGTH(plate_number) < 2")
    conn.commit()
    conn.close()


def is_vehicle_currently_parked(plate_number: str) -> bool:
    """Returns True if the vehicle currently has an ACTIVE session inside the parking lot."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id FROM parking_sessions WHERE plate_number = ? AND status = 'ACTIVE'",
        (plate_number,)
    )
    row = cursor.fetchone()
    conn.close()
    return row is not None


def verify_plate_text(ocr_text: str, confidence: float, auto_mode: bool = True) -> str:
    """
    Smart Verification:
    - Applies Malaysian plate character correction.
    - If confidence >= 75% and plate is valid (at least 2 chars), auto-confirms.
    - Otherwise prompts operator to confirm or edit.
    """
    cleaned_ocr = smart_malaysian_plate_clean(ocr_text)
    conf_ratio = confidence if confidence <= 1.0 else (confidence / 100.0)

    if auto_mode and len(cleaned_ocr) >= 2 and conf_ratio >= AUTO_CONFIRM_THRESHOLD:
        print("\n" + "=" * 60)
        print(f"  [AUTO-GATE PASS] High Confidence ({conf_ratio * 100:.1f}%) -> {cleaned_ocr}")
        print("  -> Barrier opened automatically (Zero human delay!)")
        print("=" * 60)
        return cleaned_ocr

    print("\n" + "=" * 60)
    print(f"  [LOW CONFIDENCE / MANUAL REVIEW] OCR Read: >>> {cleaned_ocr or 'UNREADABLE'} <<< ({conf_ratio * 100:.1f}%)")
    print("=" * 60)

    while True:
        user_input = input("Press [ENTER] to accept, type correct plate, or type 'CANCEL' to abort: ").strip().upper()
        if user_input == "CANCEL":
            return ""
        final_plate = cleaned_ocr if user_input == "" else normalise(user_input)
        if len(final_plate) >= 2:
            return final_plate
        print("[ERROR] Plate number cannot be empty! Please type a valid plate or 'CANCEL'.")


def handle_entry(plate_number: str, confidence: float, image_path: str) -> bool:
    """Saves a new ACTIVE parking session, blocking empty or duplicate plates."""
    if not plate_number or len(plate_number) < 2:
        print("\n[ERROR] Invalid or empty plate number. Entry aborted.")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT session_id, entry_time FROM parking_sessions WHERE plate_number = ? AND status = 'ACTIVE'",
        (plate_number,)
    )
    existing = cursor.fetchone()
    if existing:
        print(f"\n[ERROR] Vehicle {plate_number} is ALREADY inside (Entered at {existing[1]})! Duplicate entry denied.")
        conn.close()
        return False

    entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_pct = _to_pct(confidence)

    cursor.execute('''
        INSERT INTO parking_sessions (plate_number, entry_time, status, confidence, image_path)
        VALUES (?, ?, 'ACTIVE', ?, ?)
    ''', (plate_number, entry_time, conf_pct, image_path))

    conn.commit()
    conn.close()

    print(f"\n[SUCCESS] ENTRY RECORDED for {plate_number}")
    print(f"  -> Entry Timestamp : {entry_time}")
    print(f"  -> OCR Confidence  : {conf_pct:.1f}%")
    print(f"  -> Saved Photo Ref : {image_path}")
    print(f"  -> Session Status  : ACTIVE")
    return True


def handle_exit(plate_number: str) -> bool:
    """Closes an ACTIVE session when the car leaves and calculates the final fee."""
    if not plate_number or len(plate_number) < 2:
        print("\n[ERROR] Invalid or empty plate number. Exit aborted.")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT session_id, entry_time FROM parking_sessions WHERE plate_number = ? AND status = 'ACTIVE'",
        (plate_number,)
    )
    row = cursor.fetchone()

    if not row:
        print(f"\n[ERROR] No active session found for {plate_number}. Are you sure this car entered?")
        conn.close()
        return False

    session_id, entry_time = row
    exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        UPDATE parking_sessions 
        SET exit_time = ?, status = 'COMPLETED' 
        WHERE session_id = ?
    ''', (exit_time, session_id))

    conn.commit()
    conn.close()

    duration_hours, total_fee, day_type = calculate_fee(entry_time, exit_time)

    print(f"\n[SUCCESS] EXIT RECORDED for {plate_number}.")
    print(f"  -> Entered      : {entry_time} ({day_type})")
    print(f"  -> Exited       : {exit_time}")
    print(f"  -> Duration     : {duration_hours} Block(s)/Hour(s)")
    print(f"  -> Final Est Fee: RM {total_fee:.2f}")
    return True


def list_all_sessions():
    """Displays all active and completed sessions with LIVE fee calculation for active cars."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id, plate_number, entry_time, exit_time, status, confidence "
        "FROM parking_sessions WHERE LENGTH(plate_number) >= 2 ORDER BY session_id DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 95)
    print(f"{'ID':<4} | {'PLATE':<11} | {'ENTRY TIME':<19} | {'EXIT TIME':<19} | {'STATUS':<9} | {'FEE (EST)':<10} | {'CONF'}")
    print("-" * 95)
    if not rows:
        print("No parking sessions recorded yet.")
    for r in rows:
        sid, plate, ent, ext, stat, conf = r
        conf_val = _to_pct(conf)
        if stat == "ACTIVE" or not ext:
            ext_str = "--- STILL PARKED ---"
            _, live_fee, _ = calculate_fee(ent, now_str)
            fee_str = f"RM {live_fee:.2f}*"
        else:
            ext_str = ext
            _, final_fee, _ = calculate_fee(ent, ext)
            fee_str = f"RM {final_fee:.2f}"
        print(f"{sid:<4} | {plate:<11} | {ent:<19} | {ext_str:<19} | {stat:<9} | {fee_str:<10} | {conf_val:.1f}%")
    print("=" * 95)
    print("(* Note: Active sessions show live running estimated fee as of current time)")