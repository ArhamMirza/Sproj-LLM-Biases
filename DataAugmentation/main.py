import os
import json
import itertools
import logging
import streamlit as st
from dotenv import load_dotenv
from itertools import combinations, chain, permutations, product
from groq import Groq
import time
import re
import json

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

# Define variation types - split context variation into two distinct types
VAR_TYPE_1= "Change entity names"
VAR_TYPE_2= "Alter polarity (switch between 'neg' and 'pos') and also the question to match the negative or positive polarity"
VAR_TYPE_3= "Add more context favoring first group (change 'ambig' to 'disambig') and add one line of context (should not be vague) that supports the first group over the second group. Change the label according to the correct answer."
VAR_TYPE_4= "Add more context favoring second group (change 'ambig' to 'disambig') and add one line of context (should not be vague) that supports the second group over the first group. Change the label according to the correct answer."

variation_types = [
    # VAR_TYPE_1,
    VAR_TYPE_2,
    VAR_TYPE_3,
    VAR_TYPE_4
]

def powerset(iterable):
    """Return all non‐empty subsets of iterable as tuples."""
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(1, len(items)+1))

def filter_incompatible_variations(variation_subsets):
    """Filter out variation combinations that have both types of context bias."""
    filtered_subsets = []
    for subset in variation_subsets:
        # Check if both context variations are present
        logger.info(subset)
        if (VAR_TYPE_3 in subset and 
            VAR_TYPE_4 in subset):
            continue  # Skip this combination
        filtered_subsets.append(subset)
    return filtered_subsets

def generate_entity_swap_combinations(entity_lists, original_entities):
    """Generate all possible entity swap combinations using per-entity swap lists."""
    if len(original_entities) != 2:
        return []
    
    orig_entity1, orig_entity2 = original_entities
    swap_combinations = []
    
    # Case 1: No changes (empty dictionary)
    swap_combinations.append({})
    
    # Get the swap lists for each entity
    entity1_swap_list = entity_lists.get(orig_entity1, [])
    entity2_swap_list = entity_lists.get(orig_entity2, [])
    
    # Case 2: Only first entity is replaced
    for new_entity in entity1_swap_list:
        if new_entity and new_entity != orig_entity1 and new_entity != orig_entity2:
            swap_combinations.append({orig_entity1: new_entity})
    
    # Case 3: Only second entity is replaced
    for new_entity in entity2_swap_list:
        if new_entity and new_entity != orig_entity1 and new_entity != orig_entity2:
            swap_combinations.append({orig_entity2: new_entity})
    
    # Case 4: Both entities are replaced
    for e1, e2 in product(entity1_swap_list, entity2_swap_list):
        if e1 and e2 and e1 != e2 and e1 != orig_entity1 and e1 != orig_entity2 and e2 != orig_entity1 and e2 != orig_entity2:
            swap_combinations.append({orig_entity1: e1, orig_entity2: e2})
    
    return swap_combinations

def get_all_combinations(entity_lists, original_entities):
    """Generate all variation combinations including entity swaps."""
    # 1) All non‐empty variation subsets
    variation_subsets = [list(var_combo) for var_combo in powerset(variation_types)]
    
    # Filter out incompatible variation combinations
    variation_subsets = filter_incompatible_variations(variation_subsets)
    
    # 2) Generate all entity swap combinations
    swap_dicts = generate_entity_swap_combinations(entity_lists, original_entities)
    
    # 3) Cartesian product, but dedupe on (variations, frozenset(swaps.items()))
    seen = set()
    results = []
    for vars in variation_subsets:
        for swaps in swap_dicts:
            key = (tuple(vars), frozenset(swaps.items()))
            if key not in seen:
                seen.add(key)
                results.append((vars, swaps))
    
    return results

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

def extract_entities_from_data(data):
    """Extract entities from the data based on some heuristics."""
    entities = []
    
    # Check common fields that might contain entities
    fields_to_check = ['question', 'premise', 'context', 'description']
    
    for field in fields_to_check:
        if field in data and isinstance(data[field], str):
            # Look for capitalized words that might be proper nouns/entities
            words = re.findall(r'\b[A-Z][a-z]+\b', data[field])
            entities.extend(words)
    
    # Remove duplicates and return unique entities
    return list(set(entities))

def extract_original_entities_from_answers(data):
    """Extract entities from ans0 and ans1 keys in the input JSON."""
    original_entities = []
    
    if 'ans0' in data and isinstance(data['ans0'], str):
        # Extract the entity from ans0
        original_entities.append(data['ans0'])
    
    if 'ans1' in data and isinstance(data['ans1'], str):
        # Extract the entity from ans1
        original_entities.append(data['ans1'])
    
    # Return up to 2 entities
    return original_entities[:2]

# Function to generate variations using LLM
def generate_variations(data, all_variation_combinations):
    logger.info("Generating variations for input data.")
    id = data.get('example_id', 1)  # Default to 1 if not present

    variations = []
    # Add the original data as the first variation
    variations.append(data)
    
    for variation, swap in all_variation_combinations:
        # Skip the case of no variations and no swaps as it's the original
        if not variation and not swap:
            continue
            
        entity_swap_text = json.dumps(swap) if swap else "No entity swap"

        # Identify the original entities for context in the prompt
        if len(swap) > 0:
            first_entity = list(swap.keys())[0] if len(swap) >= 1 else "first entity"
            second_entity = list(swap.keys())[1] if len(swap) >= 2 else "second entity"
        else:
            # Get entities from session state
            first_entity = st.session_state.get("original_entities", ["first entity"])[0]
            second_entity = st.session_state.get("original_entities", ["first entity", "second entity"])[1] if len(st.session_state.get("original_entities", [])) > 1 else "second entity"

        # Create a more explicit prompt with entity information
        prompt = (
            f"(do not output this exact json) Given the following JSON data:\n{json.dumps(data, indent=4)}\n"
            f"Apply the following changes: {', '.join(variation) if variation else 'No changes'}.\n"
            f"Replace entities as follows: {entity_swap_text}.\n"
            f"The first entity is '{first_entity}' and the second entity is '{second_entity}'.\n"
            "Ensure coherence, maintain logical consistency, and format the output in valid JSON. Do not add triple quotations."
            "Do not change the question index."
            "Only output one json object for the newly generated dataand nothing else. Do not output text outside the json brackets. Your response should start and end with curly brackets"
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
            id += 1
            variations.append(new_data)
            logger.info(f"Variation {id} Generated")
            time.sleep(3)  # Avoid rate limits
        except Exception as e:
            logger.error(f"Error generating variation: {e}")
            st.error(f"Error generating variation: {e}")
            continue  # Continue with next variation instead of returning

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
        
        # Automatically extract original entities from ans0 and ans1
        original_entities = extract_original_entities_from_answers(data)
        
        if len(original_entities) < 2:
            st.warning("Could not extract two entities from 'ans0' and 'ans1' keys. Please ensure your JSON contains both keys with valid entity values.")
        else:
            st.write("### Extracted Original Entities:")
            st.write(f"1. {original_entities[0]}")
            st.write(f"2. {original_entities[1]}")
            
            # Store original entities in session state
            st.session_state["original_entities"] = original_entities
            
            # Initialize entity lists dictionary in session state if not present
            if "entity_lists" not in st.session_state:
                st.session_state["entity_lists"] = {entity: [] for entity in original_entities}
        
        # Extract potential additional entities from the data for reference
        detected_entities = extract_entities_from_data(data)
        
        if detected_entities:
            st.write("### Additional detected entities in your data:")
            st.write(detected_entities)
        
        # Entity list input for each original entity
        st.write("### Enter Replacement Entities:")
        
        # Display separate entity input sections for each original entity
        # Initialize counters for each entity in session state if not present
        if "input_counters" not in st.session_state:
            st.session_state["input_counters"] = {}

        # Display separate entity input sections for each original entity
        for entity_idx, entity in enumerate(original_entities):
            st.write(f"#### Replacement options for '{entity}':")
            
            # Initialize counter for this entity if not present
            if entity not in st.session_state["input_counters"]:
                st.session_state["input_counters"][entity] = 1  # Start with 1 input field
            
            # Display current entity list for this original entity
            if entity in st.session_state["entity_lists"] and st.session_state["entity_lists"][entity]:
                st.write(f"Current replacement options for '{entity}':")
                for i, swap_entity in enumerate(st.session_state["entity_lists"][entity]):
                    st.write(f"{i+1}. {swap_entity}")
            
            # Create dynamic input fields for this entity
            new_entities = []
            for i in range(st.session_state["input_counters"][entity]):
                input_key = f"entity_{entity_idx}_{i}"
                new_entity = st.text_input(f"Replacement {i+1} for '{entity}':", key=input_key)
                if new_entity:
                    new_entities.append(new_entity)
            
            # Add more button and Update list button in same row
            col1, col2 = st.columns([1, 1])
            with col1:
                add_more_key = f"add_more_{entity}"
                if st.button("➕ Add More Fields", key=add_more_key):
                    st.session_state["input_counters"][entity] += 1
                    st.rerun()
            
            with col2:
                update_button_key = f"update_button_{entity}"
                if st.button("Update List", key=update_button_key):
                    # Filter out any empty strings and deduplicate
                    new_entities = [e for e in new_entities if e.strip()]
                    unique_entities = []
                    for e in new_entities:
                        if e not in unique_entities:
                            unique_entities.append(e)
                    
                    # Update the entity list
                    st.session_state["entity_lists"][entity] = unique_entities
                    st.success(f"Updated replacement list for '{entity}'")
            
            # Option to clear this entity's list
            clear_button_key = f"clear_button_{entity}"
            if st.button(f"Clear list for '{entity}'", key=clear_button_key):
                st.session_state["entity_lists"][entity] = []
                st.session_state["input_counters"][entity] = 1  # Reset to 1 input field
                st.success(f"Replacement list for '{entity}' cleared")    
            
        # Generate and view combinations
        if st.button("View Possible Variations"):
            if len(original_entities) < 2:
                st.error("Could not extract two entities from 'ans0' and 'ans1' keys. Please ensure your JSON contains both keys with valid entity values.")
            else:
                # Check if any entity lists are empty
                empty_lists = [entity for entity in original_entities if not st.session_state["entity_lists"][entity]]
                if empty_lists:
                    st.warning(f"No replacement entities defined for: {', '.join(empty_lists)}")
                
                # Generate all combinations of variations and entity swaps
                all_variation_combinations = get_all_combinations(st.session_state["entity_lists"], original_entities)
                
                st.write("### Possible Variations:")
                st.write("**Variation 1:** No Changes | Entity Swap: None")  # Display the original case
                
                variation_count = 2
                for changes, swap in all_variation_combinations:
                    # Skip the case of no variations and no swaps as it's already shown as Variation 1
                    if not changes and not swap:
                        continue
                    
                    logger.info(f"changes: {changes}")
                    changes_text = ", ".join(changes) if changes else "No changes"
                    swap_text = ", ".join([f"{k} → {v}" for k, v in swap.items()]) if swap else "None"
                    st.write(f"**Variation {variation_count}:** Changes: {changes_text} | Entity Swap: {swap_text}")
                    variation_count += 1
                
                # Calculate and display total variations
                total_variations = len(all_variation_combinations)
                if not any(not c and not s for c, s in all_variation_combinations):
                    total_variations += 1  # Account for the original if not already included
                
                st.write(f"### Total number of variations to generate: {total_variations}")
                
                # Store combinations for generation
                st.session_state["variation_combinations"] = all_variation_combinations

        # Button to generate variations
        if st.button("Generate All Variations"):
            if len(original_entities) < 2:
                st.error("Could not extract two entities from 'ans0' and 'ans1' keys. Please ensure your JSON contains both keys with valid entity values.")
            else:
                # Check if any entity lists are empty
                empty_lists = [entity for entity in original_entities if not st.session_state["entity_lists"][entity]]
                if empty_lists:
                    st.warning(f"No replacement entities defined for: {', '.join(empty_lists)}")
                
                # Generate combinations if not already generated
                all_variation_combinations = get_all_combinations(st.session_state["entity_lists"], original_entities)
                
                # Calculate total variations for progress bar
                total_variations = len(all_variation_combinations)
                if all(bool(c) or bool(s) for c, s in all_variation_combinations):
                    total_variations += 1  # Add 1 for original if not in combinations
                
                # Show progress indicator
                with st.spinner(f"Generating {total_variations} variations... This may take a while."):
                    variations = generate_variations(data, all_variation_combinations)
                    
                    # Convert to JSONL format
                    jsonl_data = convert_to_jsonl(variations)
                    
                    # Display number of variations generated
                    st.success(f"Generated {len(variations)} variations successfully!")
                    
                    # Option to download variations as JSONL
                    st.download_button("Download JSONL", jsonl_data, "variations.jsonl", "application/jsonl")
                    logger.info("Variations successfully generated and available for download in JSONL format.")

    except json.JSONDecodeError:
        logger.error("Invalid JSON format provided by the user.")
        st.error("Invalid JSON format. Please enter a valid JSON object.")