## Import the necessary modules
import json
import ollama

## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result

# Paths / constants
ITEMS_FILE = "items.json"
OUTPUT_FILE = "output/match_result.json"
MODEL_NAME = "qwen2.5:1.5b"


## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant. Your job is to match a user's "
        "description of a lost item against a database of found items.\n\n"
        "RULES:\n"
        "1. Use ONLY the items provided in the available_items JSON list.\n"
        "2. Not all details must match — partial matches count.\n"
        "3. Return ONLY valid JSON, with exactly this structure:\n"
        '   { "matches": ["ITEM_ID"], "confidence": "LOW" }\n'
        '4. "matches" must contain the IDs of all possible matching items.\n'
        '5. "confidence" must be exactly one of: LOW, MEDIUM, HIGH.\n'
        "6. If no item matches, return an empty list: "
        '{ "matches": [], "confidence": "LOW" }\n'
        "7. Do not include any explanation, markdown, or extra text — "
        "output raw JSON only."
    )

    available_items_json = json.dumps(available_items, indent=2)
    user_prompt = (
        f"A user lost the following item:\n"
        f'"{description}"\n\n'
        f"Here are the available found items (JSON):\n"
        f"{available_items_json}\n\n"
        f"Return the JSON result now."
    )
    return system_prompt, user_prompt


## Logic to ask Qwen for all the possible matches.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result.
def parse_response(response_text):
    text = response_text.strip()
    # 剥离可能的 markdown 代码围栏
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


## Logic to validate the result returned by Qwen.
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False

    matches = result["matches"]
    confidence = result["confidence"]

    if not isinstance(matches, list):
        return False
    if not all(isinstance(m, str) for m in matches):
        return False
    if confidence not in ("LOW", "MEDIUM", "HIGH"):
        return False

    valid_ids = {item["id"] for item in available_items}
    for match_id in matches:
        if match_id not in valid_ids:
            return False

    return True


## Logic to display the matches found by Qwen in a user-friendly format.
def display_matches(result, available_items):
    print("\nMATCH RESULT")
    print("-" * 50)
    print(f"Confidence: {result['confidence']}")
    print()

    matches = result.get("matches", [])

    if not matches:
        print("No matches were found.")
        print("Possible matches: []")
        return

    print("Possible matches:")
    items_by_id = {item["id"]: item for item in available_items}
    for match_id in matches:
        item = items_by_id.get(match_id)
        if not item:
            continue
        print()
        print(f"ID: {item['id']}")
        print(f"Item: {item.get('item', 'N/A')}")
        print(f"Color: {item.get('color', 'N/A')}")
        print(f"Location: {item.get('location', 'N/A')}")
        print(f"Date found: {item.get('date_found', 'N/A')}")


## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50)
    print()

    description = input("Describe the item you lost: ").strip()

    items = load_items(ITEMS_FILE)
    available_items = get_unclaimed_items(items)

    system_prompt, user_prompt = build_prompt(description, available_items)

    print("\nSearching for possible matches...")

    response_text = ask_qwen(system_prompt, user_prompt)
    result = parse_response(response_text)

    if not validate_result(result, available_items):
        print("\n[Error] The model returned an invalid result.")
        print("Raw response:", response_text)
        return

    display_matches(result, available_items)

    save_result(result, OUTPUT_FILE)
    print(f"\nResult saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()