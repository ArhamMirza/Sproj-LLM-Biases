import os
import json
import itertools
import logging
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is missing. Ensure it is set in the environment variables.")
    st.error("API Key not found. Please set your API Key.")
    st.stop()

# Initialize Groq API client
groq = Groq(api_key=GROQ_API_KEY)
MODEL = "llama3-70b-8192"

# Streamlit UI
st.set_page_config(page_title="Data Variation Generator", layout="wide")
st.title("🔀 Data Variation Generator using Groq LLM")

# Define variation types
variation_types = [
    "Change entity names",
    "Alter polarity (switch between 'neg' and 'pos') and also the question to match the negative or positive polarity",
    "Add more context (change 'ambig' to 'disambig') and add one line of context (should not be vague), giving information that supports one group over the other",
]

# Function to generate all possible combinations
def get_all_combinations(entity_swaps):
    all_combinations = []
    seen_combinations = set()

    variation_count = len(variation_types)
    
    # Generate all combinations of variation_types
    for i in range(1, 2 ** variation_count):  # Skip the empty set (i = 0)
        combination = [variation_types[j] for j in range(variation_count) if (i & (1 << j))]
        
        # Skip invalid cases
        if "Change entity names" in combination:
            continue

        combo_key = (tuple(combination), frozenset())
        if combo_key not in seen_combinations:
            seen_combinations.add(combo_key)
            logger.info(combination)
            all_combinations.append((combination, {}))

        # Generate all non-empty combinations of entity swaps
        swap_count = len(entity_swaps)
        for swap_mask in range(1, 2 ** swap_count):  # Non-empty subsets of entity swaps
            merged_swaps = {}
            for k in range(swap_count):
                if swap_mask & (1 << k):  # If the k-th swap is included
                    merged_swaps.update(entity_swaps[k])

            combo_key = (tuple(combination), frozenset(merged_swaps.items()))
            if combo_key not in seen_combinations:
                seen_combinations.add(combo_key)
                all_combinations.append((combination, merged_swaps))

    return all_combinations


import re
import json

def extract_valid_json(response_text):
    """Extracts and returns valid JSON as a string."""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)  # Extract JSON content
        if json_match:
            json_str = json_match.group(0)  # Get matched JSON as string
            return json_str  # Return JSON string
        else:
            return None  # No valid JSON found
    except Exception as e:
        return None  # Return None if parsing fails

# Function to generate variations using LLM
def generate_variations(data, all_variation_combinations):
    logger.info("Generating variations for input data.")
    id = data['example_id']

    variations = []
    for variation, swap in all_variation_combinations:
        entity_swap_text = json.dumps(swap) if swap else "No entity swap"

        prompt = (
            f"Given the following JSON data:\n{json.dumps(data, indent=4)}\n"
            f"Apply the following changes: {', '.join(variation) if variation else 'No changes'}.\n"
            f"Replace entities as follows: {entity_swap_text}.\n"
            "Ensure coherence, maintain logical consistency, and format the output in valid JSON. Do not add triple quotations."
        )

        try:
            response = groq.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an AI that generates structured data variations."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096
            )
            response_json = response.choices[0].message.content
            logger.info(response_json)

            new_data = json.loads(extract_valid_json(response_json))
            new_data['example_id'] = id
            id+=1
            variations.append(new_data)
            logger.info(f"Variation {id} Generated")
            time.sleep(1)  # Avoid rate limits
        except Exception as e:
            logger.error(f"Error generating variation: {e}")
            st.error(f"Error generating variation: {e}")
            return variations

    return variations

def convert_to_jsonl(variations):
    return "\n".join(json.dumps(variation) for variation in variations)

# User input for JSON object
json_input = st.text_area("Enter JSON object:")

if json_input:
    try:
        data = json.loads(json_input)
        logger.info("User provided valid JSON input.")
        st.write("### Original Data:", data)

        # Entity Swap Input
        if "entity_swaps" not in st.session_state:
            st.session_state["entity_swaps"] = []

        st.write("### Enter Entity Replacements:")
        new_entity = st.text_input("Entity to Replace:",autocomplete="off")
        replacement = st.text_input("Replace with:",autocomplete="off")

        if st.button("Add Entity Swap"):
            if new_entity and replacement:
                st.session_state["entity_swaps"].append({new_entity: replacement})
                st.success(f"Added: '{new_entity}' → '{replacement}'")

        # Display added entity swaps
        if st.session_state["entity_swaps"]:
            st.write("#### Current Entity Swaps:")
            for swap in st.session_state["entity_swaps"]:
                st.write(swap)

        # Generate all combinations
        all_variation_combinations = get_all_combinations(st.session_state["entity_swaps"])

        # Button to view all possible combinations
        if st.button("View Combinations"):
            st.write("### Possible Combinations:")
            st.write("**Combination 1:** No Changes | Entity Swap: None")  # Display the original case
            for i, (changes, swap) in enumerate(all_variation_combinations, start=2):
                st.write(f"**Combination {i}:** Changes: {', '.join(changes)} | Entity Swap: {swap if swap else 'None'}")

        # Button to generate variations
        if st.button("Generate Variations"):
            variations = generate_variations(data, all_variation_combinations)
            # Convert to JSONL format
            jsonl_data = convert_to_jsonl(variations)

            # Option to download variations as JSONL
            st.download_button("Download JSONL", jsonl_data, "variations.jsonl", "application/jsonl")
            logger.info("Variations successfully generated and available for download in JSONL format.")

    except json.JSONDecodeError:
        logger.error("Invalid JSON format provided by the user.")
        st.error("Invalid JSON format. Please enter a valid JSON object.")
