import requests
import pandas as pd
import traceback
import json
from datetime import datetime, timezone

class FlightDataFetcher:
    """
    Fetches recent flight data incrementally.
    Includes extra timestamp debugging.
    """
    def __init__(self, base_url, api_endpoint="/api/flights/recent", fetch_limit=200):
        self.base_url = base_url
        self.api_endpoint = api_endpoint
        self.full_url = self.base_url.rstrip('/') + '/' + self.api_endpoint.lstrip('/')
        self.fetch_limit = fetch_limit
        self.last_processed_timestamp = None # Initialize state
        self.initial_fetch_done = False # Track initial fetch explicitly
        print(f"[Fetcher Init] URL: {self.full_url}, Limit: {self.fetch_limit}")

    def fetch_next_batch(self):
        params = {'limit': self.fetch_limit}
        current_last_ts = self.last_processed_timestamp # Store for comparison later
        request_desc = f"Fetching records (limit={self.fetch_limit})"

        if current_last_ts:
            params['since'] = current_last_ts
            request_desc = f"Fetching records since {current_last_ts} (limit={self.fetch_limit})"
        else:
            request_desc = f"Performing initial fetch (limit={self.fetch_limit})"
            print("[Fetcher Debug] No last_processed_timestamp, performing initial fetch.")

        print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] [Fetcher] {request_desc}...")
        print(f"[Fetcher Debug] Request Params: {params}")

        try:
            response = requests.get(self.full_url, params=params, timeout=30)
            print(f"[Fetcher Debug] API Response Status: {response.status_code}")
            response.raise_for_status()
            data = response.json()

            if data:
                print(f"    [Fetcher] Fetched {len(data)} records.")
                self.initial_fetch_done = True # Mark that we received something at least once
                newest_timestamp_in_batch = None
                try:
                    # Find the maximum 'updated_at' timestamp in the received batch
                    timestamps = [item['updated_at'] for item in data if 'updated_at' in item and item['updated_at']]
                    if timestamps:
                        newest_timestamp_in_batch = max(timestamps)
                        print(f"    [Fetcher Debug] Newest timestamp found in batch: {newest_timestamp_in_batch}")

                        # --- Timestamp Update Logic ---
                        # Only update if the new timestamp is different and actually newer
                        # Use "1970..." as a safe starting point for comparison if current_last_ts is None
                        compare_ts = current_last_ts or "1970-01-01T00:00:00Z"
                        if newest_timestamp_in_batch > compare_ts:
                            print(f"    [Fetcher] SUCCESS: New timestamp {newest_timestamp_in_batch} > {compare_ts}. Updating state.")
                            self.last_processed_timestamp = newest_timestamp_in_batch
                        else:
                            print(f"    [Fetcher Debug] SKIPPING state update: Newest timestamp {newest_timestamp_in_batch} is NOT > {compare_ts}")
                            # This is likely why subsequent fetches find nothing if timestamps repeat or aren't strictly increasing

                    else:
                        print("    [Fetcher] Warning: Received data but no valid 'updated_at' timestamps found. State not updated.")
                except Exception as e:
                    print(f"    [Fetcher] Error processing timestamps: {e}. State not updated.")

                df = pd.DataFrame(data)
                # Add minimal checks/conversions if needed
                return df
            else:
                # API returned empty list []
                if not self.initial_fetch_done and current_last_ts is None:
                     print(f"    [Fetcher] No records returned on INITIAL fetch.")
                else:
                     print(f"    [Fetcher] No NEW records returned by API since {current_last_ts}.")
                self.initial_fetch_done = True # Mark initial attempt done even if empty
                return pd.DataFrame() # Return empty DataFrame

        # --- Error Handling ---
        except requests.exceptions.ConnectionError as e: print(f"\n!!! [Fetcher] Connection Error: {e}"); return None
        except requests.exceptions.Timeout: print(f"\n!!! [Fetcher] Timeout Error."); return None
        except requests.exceptions.RequestException as e:
            print(f"\n!!! [Fetcher] HTTP Error: {e}.")
            try: print(f"    Response Status Code: {e.response.status_code}, Body: {e.response.text[:200]}...")
            except: pass
            return None
        except json.JSONDecodeError as e:
            print(f"\n!!! [Fetcher] JSON Decode Error: {e}.")
            try: print(f"    Raw Response Text: {response.text[:200]}...")
            except: pass
            return None
