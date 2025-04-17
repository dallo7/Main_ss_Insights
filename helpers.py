from openai import OpenAI


import pandas as pd

# def analyze_flight_data(callsign, flight_number):
#
#     YOUR_API_KEY="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"
#
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are an expert in aviation data analysis. Your task is to determine the airline name and the nature of a flight "
#                 "based on the provided callsign and flight number. You should follow industry standards and common aviation patterns "
#                 "to classify the flight accurately. Your response must be clear, structured, and formatted as plain text."
#             ),
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Input:\nCallsign: {callsign}\nFlight: {flight_number}\n\n"
#                 "Task: Identify the airline and the nature of the flight based on the given details.\n\n"
#                 "Guidelines:\n"
#                 "1. Determine the airline name using industry-standard callsign prefixes:\n"
#                 "    - THY / TK → Turkish Airlines\n"
#                 "    - ET → Ethiopian Airlines\n"
#                 "    - RED / HB → Red Cross (often medical/humanitarian flights)\n"
#                 "    - BDR → BADR Airlines\n"
#                 "    - CKS → Kalitta Air (Cargo)\n"
#                 "    - KLM → KLM Royal Dutch Airlines\n"
#                 "    - AFR → Air France\n"
#                 "    - JAL → Japan Airlines\n\n"
#                 "2. Identify the flight nature based on standard aviation classifications:\n"
#                 "    - PASSENGER → Scheduled commercial passenger flights\n"
#                 "    - CARGO → Dedicated freight flights\n"
#                 "    - CHARTER → Private or specially arranged flights\n"
#                 "    - FERRY → Non-revenue repositioning flights\n"
#                 "    - MEDIVAC → Medical evacuation flights (e.g., Red Cross, emergency medical services)\n"
#                 "    - MILITARY → Defense-related or government-operated flights\n\n"
#                 "Examples:\n\n"
#                 "Example 1:\n"
#                 "Input:\n"
#                 "Callsign: THY613\n"
#                 "Flight: TK613\n"
#                 "Output:\n"
#                 "Company Name: Turkish Airlines\n"
#                 "Flight Nature: PASSENGER\n\n"
#                 "Example 2:\n"
#                 "Input:\n"
#                 "Callsign: RED912\n"
#                 "Flight: HB-LTG\n"
#                 "Output:\n"
#                 "Company Name: Red Cross\n"
#                 "Flight Nature: MEDIVAC\n\n"
#                 "Example 3:\n"
#                 "Input:\n"
#                 "Callsign: CKS204\n"
#                 "Flight: CKS204\n"
#                 "Output:\n"
#                 "Company Name: Kalitta Air\n"
#                 "Flight Nature: CARGO\n\n"
#                 "Output Format:\n"
#                 "Company Name: [Determined Airline Name]\n"
#                 "Flight Nature: [Determined Flight Nature]\n"
#             ),
#         }
#     ]
#
#     client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")
#     response = client.chat.completions.create(
#         model="sonar-pro",
#         messages=messages,
#     )
#
#     extracted_data = response.choices[0].message.content
#
#     return extracted_data



from openai import OpenAI

def insights_for_flight_data(data):

    YOUR_API_KEY = "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"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a master Airplane live feeds analyst specialized in South Sudan airspace monitoring. "
                "Your task is to filter and analyze flight data to identify aircraft passing through or landing in South Sudan airspace. "
                "Provide accurate, actionable insights for the operations team focused on flight patterns and airspace utilization."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Dataset:\n{data}\n\n"
                "### Tasks ###\n"
                "1. **Filter flights**: Return only flights that pass through, land in, or take off from South Sudan airspace.\n"
                "   - **Limit output to 5 rows**.\n"
                "   - **Include only these columns**: [flight_id, latitude, longitude, operator, registration, time, "
                "origin, destination, altitude, aircraft_type].\n"
                "Ignore all UN Flights and Redcross Flights. \n "
                "   - **Check origin and destination fields** to confirm flights Not landing in or departing from South Sudan.\n"
                "\n"
                "2. **Analyze and provide Any 7 key insights, you can default to the below if you do not have anything interesting**:\n"
                "\n"
                "3. **Output must be valid JSON with the following strict format**:"
                "{\n"
                "    \"filtered_flights\": [\n"
                "        { \"flight_id\": \"ABC123\", \"latitude\": \"7.6892\", \"longitude\": \"28.5212\", \"operator\": \"Ethiopian Airlines\", \"registration\": \"ET-AKF\", \"time\": \"2025-03-29 14:20:30\", \"origin\": \"Addis Ababa\", \"destination\": \"Juba\", \"altitude\": \"38000\", \"aircraft_type\": \"B787\" },\n"
                "        ... (up to 5 rows) ...\n"
                "    ],\n"
                "    \"insights\": [\n"
                "        { \"title\": \"Flights Landing/Taking Off\", \"description\": \"...\" },\n"
                "        { \"title\": \"Common Routes\", \"description\": \"...\" },\n"
                "        { \"title\": \"Regular Operators\", \"description\": \"...\" },\n"
                "        { \"title\": \"Peak Times\", \"description\": \"...\" },\n"
                "        { \"title\": \"Altitude Patterns\", \"description\": \"...\" },\n"
                "    ]\n"
                "}\n"
            ),
        }
    ]
    client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )

    return response.choices[0].message.content

# print(insights_for_flight_data("https://cdn.wsform.com/wp-content/uploads/2020/06/industry.csv"))


import requests
import json

def generate_and_print_cleaned_gemini_output(prompt):

    api_key = "AIzaSyBdPf2vNe06pjA6MKtTATtz3g3-pQZigqo"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a master Airplane live feeds analyst specialized in South Sudan airspace monitoring. "
                "Your task is to filter and analyze flight data to identify aircraft passing through or landing in South Sudan airspace. "
                "Provide accurate, actionable insights for the operations team focused on flight patterns and airspace utilization."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Dataset:\n{prompt}\n\n"
                "### Tasks ###\n"
                "1. **Filter flights**: Return only flights that pass through, land in, or take off from South Sudan airspace.\n"
                "   - **Limit output to 5 rows**.\n"
                "   - **Include only these columns**: [flight_id, latitude, longitude, operator, registration, time, "
                "origin, destination, altitude, aircraft_type].\n"
                "   - **Check origin and destination fields** to confirm flights landing in or departing from South Sudan.\n"
                "\n"
                "2. **Analyze and provide Any 7 key insights, you can default to the below if you do not have anything interesting**:\n"
                "   - **Flights landing/taking off**: Identify flights with South Sudan as their origin or destination.\n"
                "   - **Common routes**: Highlight frequent routes passing through South Sudan airspace.\n"
                "   - **Regular operators**: Which airlines frequently use South Sudan airspace.\n"
                "   - **Peak times**: When is South Sudan airspace most congested.\n"
                "   - **Altitude patterns**: Common flight levels when crossing South Sudan.\n"
                "\n"
                "3. **Output must be valid JSON with the following strict format**:"
                "{\n"
                "    \"filtered_flights\": [\n"
                "        { \"flight_id\": \"ABC123\", \"latitude\": \"7.6892\", \"longitude\": \"28.5212\", \"operator\": \"Ethiopian Airlines\", \"registration\": \"ET-AKF\", \"time\": \"2025-03-29 14:20:30\", \"origin\": \"Addis Ababa\", \"destination\": \"Juba\", \"altitude\": \"38000\", \"aircraft_type\": \"B787\" },\n"
                "        ... (up to 5 rows) ...\n"
                "    ],\n"
                "    \"insights\": [\n"
                "        { \"title\": \"Flights Landing/Taking Off\", \"description\": \"3 flights had South Sudan as their origin or destination during the analyzed period.\" },\n"
                "        { \"title\": \"Common Routes\", \"description\": \"The route between Addis Ababa and Juba was the most frequent, with 5 flights observed.\" },\n"
                "        { \"title\": \"Regular Operators\", \"description\": \"Ethiopian Airlines was the most frequent operator, accounting for 60% of observed flights.\" },\n"
                "        { \"title\": \"Peak Times\", \"description\": \"The busiest period for South Sudan airspace was between 10:00 AM and 12:00 PM local time.\" },\n"
                "        { \"title\": \"Altitude Patterns\", \"description\": \"Most flights crossed South Sudan airspace at altitudes between 37,000 and 39,000 feet.\" },\n"
                "    ]\n"
                "}\n"
            ),
        }
    ]

    data = {
        "contents": [{
            "parts": [{"text": messages}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        api_response = response.json()

        # Attempt to extract the generated text
        if 'candidates' in api_response and api_response['candidates']:
            if 'content' in api_response['candidates'][0] and 'parts' in api_response['candidates'][0]['content']:
                generated_text = ""
                for part in api_response['candidates'][0]['content']['parts']:
                    if 'text' in part:
                        generated_text += part['text']
                cleaned_output = {'generated_text': generated_text.strip()}
                print("Cleaned Output (Dictionary):")
                print(cleaned_output)
                print("\nGenerated Text:")
                print(cleaned_output['generated_text'])
                return cleaned_output
            else:
                print("Failed to find 'content' or 'parts' in the API response.")
                return None
        else:
            print("Failed to find 'candidates' in the API response.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        return None