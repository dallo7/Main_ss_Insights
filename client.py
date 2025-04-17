import logging
import csv
import json
import zlib
import os
import sys
import time
import threading
from datetime import datetime, timezone
from amqp_consumer import AMQPConsumer


# --- Configuration ---
CONNECTION_STRING = "Endpoint=sb://fr24-position-feed-7.servicebus.windows.net/;SharedAccessKeyName=capitalpay-consumer;SharedAccessKey=Xx/TX47FHiNDVizW8pmS563oc0SOVHI3l+AEhPWaudw=;EntityPath=capitalpay"
CONSUMER_GROUP = os.environ.get('EVENT_HUB_CONSUMER_GROUP', '$Default')
STORAGE_CONNECTION_STR = os.environ.get('STORAGE_CONNECTION_STR', None)
BLOB_CONTAINER_NAME = os.environ.get('BLOB_CONTAINER_NAME', None)
PROXY_HOSTNAME = os.environ.get('PROXY_HOSTNAME', None)
PROXY_PORT = os.environ.get('PROXY_PORT', None)
PROXY_USER = os.environ.get('PROXY_USER', None)
PROXY_PASS = os.environ.get('PROXY_PASS', None)
CLIENTNAME = os.environ.get('CLIENTNAME', 'python-example-consumer-append')
DNS_CHECK_INTERVAL = int(os.environ.get('DNS_CHECK_INTERVAL', 15))

# --- CSV Output Configuration ---
CSV_OUTPUT_DIR = 'flight_data_output'  # Directory for the output file
OUTPUT_CSV_FILENAME = 'flight_data_live.csv'  # Fixed filename
FULL_CSV_PATH = os.path.join(CSV_OUTPUT_DIR, OUTPUT_CSV_FILENAME)
CSV_WRITE_INTERVAL_SECONDS = 20
# Define consistent CSV headers
CSV_FIELDNAMES = [
    'flight_id', 'addr', 'lat', 'lon', 'track', 'alt', 'speed',
    'squawk', 'radar_id', 'model', 'reg', 'last_update', 'origin',
    'destination', 'flight', 'on_ground', 'vert_speed', 'callsign',
    'source_type', 'eta', 'enhanced_json'
]

# Global variables for accumulating records (thread-safe)
accumulated_records = []
accumulated_lock = threading.Lock()

print(FULL_CSV_PATH)

# --- Helper Functions ---

def read_content(gzip_data):
    """
    Decompress incoming gzip data, return parsed JSON content. Handles errors.
    (Implementation unchanged from previous refined version)
    """
    try:
        json_data = zlib.decompress(gzip_data)
        content = json.loads(json_data)
        return content
    except zlib.error as e:
        print(f"Error decompressing data: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data: {e}")
    except Exception as e:
        print(f"Unexpected error reading content: {e}")
    return None


def inspect_flight(flight_id, values):
    """
    Convert flight's list of values into a dictionary. Handles 'enhanced' as JSON.
    Returns None for heartbeat messages. Ensures all standard fields exist.
    (Implementation unchanged from previous refined version)
    """
    if flight_id == "0":
        return None

    base_names = [
        'addr', 'lat', 'lon', 'track', 'alt', 'speed',
        'squawk', 'radar_id', 'model', 'reg', 'last_update', 'origin',
        'destination', 'flight', 'on_ground', 'vert_speed', 'callsign',
        'source_type', 'eta'
    ]
    flight_record = {'flight_id': flight_id}
    for i, name in enumerate(base_names):
        flight_record[name] = values[i] if i < len(values) else None

    if len(values) > len(base_names):
        flight_record['enhanced_json'] = json.dumps(values[len(base_names)])
    else:
        flight_record['enhanced_json'] = json.dumps({})

    for field in CSV_FIELDNAMES:
        if field not in flight_record:
            flight_record[field] = None
    return flight_record


# --- CSV Writing Logic (Modified for Appending) ---

def append_batch_to_csv(records_to_write, filename, headers):
    """
    Appends a list of flight record dictionaries to a single CSV file.
    Writes the header only if the file is new or empty.
    """
    if not records_to_write:
        print(f"[{datetime.now(timezone.utc).isoformat()}] No records accumulated to append.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if file exists and is empty to determine if header is needed
    file_exists = os.path.isfile(filename)
    write_header = not file_exists or os.path.getsize(filename) == 0

    print(f"[{datetime.now(timezone.utc).isoformat()}] Appending {len(records_to_write)} records to {filename}...")
    try:
        # Open file in append mode ('a')
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')

            if write_header:
                writer.writeheader()
                print(f"[{datetime.now(timezone.utc).isoformat()}] Wrote header to {filename}.")

            writer.writerows(records_to_write)  # Append the new rows
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] Successfully appended {len(records_to_write)} records to {filename}")

    except IOError as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] Error appending to CSV file {filename}: {e}")
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] Unexpected error during CSV append: {e}")


def csv_writer_thread_func():
    """
    Background thread function that periodically appends accumulated records to the single CSV file.
    """
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] CSV Append Writer thread started. Will write every {CSV_WRITE_INTERVAL_SECONDS} seconds to {FULL_CSV_PATH}.")
    while True:
        time.sleep(CSV_WRITE_INTERVAL_SECONDS)
        records_batch = []
        with accumulated_lock:
            if accumulated_records:
                # Copy records to process outside the lock and clear the global list
                records_batch = accumulated_records[:]
                accumulated_records.clear()
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] Cleared accumulated records for appending (Count: {len(records_batch)}).")

        # Append the copied batch (if any) outside the lock
        if records_batch:
            # Call the append function with the fixed filename and headers
            append_batch_to_csv(records_batch, FULL_CSV_PATH, CSV_FIELDNAMES)


def start_csv_writer_thread():
    """
    Starts the CSV writer function in a background daemon thread.
    (Implementation unchanged)
    """
    thread = threading.Thread(target=csv_writer_thread_func, daemon=True)
    thread.name = "CSVAppendWriterThread"  # Renamed for clarity
    thread.start()
    print(f"[{datetime.now(timezone.utc).isoformat()}] CSV Append Writer background thread initiated.")
    return thread


# --- AMQP Callback and Consumer ---

def on_receive_callback(gzip_data):
    """
    Callback function to process incoming data and accumulate records for appending.
    (Implementation unchanged from previous refined version)
    """
    content = read_content(gzip_data)
    if not content:
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] Warning: Received empty or unreadable message content. Skipping.")
        return

    content.pop('full_count', None)
    content.pop('version', None)

    if not content:
        return

    batch_added_count = 0
    for flight_id, flight_info in content.items():
        print(f"[{flight_info}]")
        record = inspect_flight(flight_id, flight_info)
        if record:
            with accumulated_lock:
                accumulated_records.append(record)
            batch_added_count += 1

    if batch_added_count > 0:
        with accumulated_lock:  # Get size under lock for accuracy
            current_size = len(accumulated_records)
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] Added {batch_added_count} records to accumulation queue (Current size: {current_size}).")


def consume_amqp(callback=on_receive_callback):
    """
    Initializes the AMQP consumer, starts the CSV append writer thread,
    and begins consuming messages.
    (Implementation largely unchanged, uses updated thread start function)
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] Initializing AMQP Consumer for CSV Appending...")
    csv_thread = start_csv_writer_thread()  # Starts the append writer thread

    consumer = None
    try:
        consumer = AMQPConsumer(
            connection_string=CONNECTION_STRING,
            consumer_group=CONSUMER_GROUP,
            storage_connection_string=STORAGE_CONNECTION_STR if STORAGE_CONNECTION_STR else None,
            blob_container_name=BLOB_CONTAINER_NAME if BLOB_CONTAINER_NAME else None,
            proxy_host=PROXY_HOSTNAME if PROXY_HOSTNAME else None,
            proxy_port=int(PROXY_PORT) if PROXY_PORT else None,
            proxy_user=PROXY_USER if PROXY_USER else None,
            proxy_pass=PROXY_PASS if PROXY_PASS else None,
            dns_check_interval=DNS_CHECK_INTERVAL
        )
        consumer.set_callback(callback)
        print(f"[{datetime.now(timezone.utc).isoformat()}] Starting AMQP message consumption...")
        consumer.consume()
        print(f"[{datetime.now(timezone.utc).isoformat()}] AMQP consumer finished gracefully.")

    except KeyboardInterrupt:
        print(f"\n[{datetime.now(timezone.utc).isoformat()}] KeyboardInterrupt received. Stopping consumer...")
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] An unexpected error occurred during AMQP consumption: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[{datetime.now(timezone.utc).isoformat()}] Main consumer function exiting.")
        # --- Optional: Write remaining records on exit ---
        # This ensures data received just before shutdown is saved.
        print(f"[{datetime.now(timezone.utc).isoformat()}] Writing any remaining accumulated records before exit...")
        final_records = []
        with accumulated_lock:
            if accumulated_records:
                final_records = accumulated_records[:]
                accumulated_records.clear()
        if final_records:
            append_batch_to_csv(final_records, FULL_CSV_PATH, CSV_FIELDNAMES)
        # ------------------------------------------------


# --- Main execution ---
if __name__ == "__main__":
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting Flight Data Processor (Appending Mode)...")
    # Ensure the output directory exists before starting
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    consume_amqp(on_receive_callback)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Flight Data Processor finished.")
