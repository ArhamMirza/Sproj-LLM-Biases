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

def generate_entity_swap_combinations(entity_list, original_entities):
    """Generate all possible entity swap combinations including individual and paired swaps."""
    if len(original_entities) != 2:
        return []
    
    orig_entity1, orig_entity2 = original_entities
    swap_combinations = []
    
    # Case 1: No changes (empty dictionary)
    swap_combinations.append({})
    
    # Case 2: Only first entity is replaced
    for new_entity in entity_list:
        if new_entity != orig_entity1 and new_entity != orig_entity2:
            swap_combinations.append({orig_entity1: new_entity})
    
    # Case 3: Only second entity is replaced
    for new_entity in entity_list:
        if new_entity != orig_entity1 and new_entity != orig_entity2:
            swap_combinations.append({orig_entity2: new_entity})
    
    # Case 4: Both entities are replaced
    for e1, e2 in product(entity_list, entity_list):
        if e1 != e2 and e1 != orig_entity1 and e1 != orig_entity2 and e2 != orig_entity1 and e2 != orig_entity2:
            swap_combinations.append({orig_entity1: e1, orig_entity2: e2})
    
    return swap_combinations

def get_all_combinations(entity_list, original_entities):
    """Generate all variation combinations including entity swaps."""
    # 1) All non‐empty variation subsets
    variation_subsets = [list(var_combo) for var_combo in powerset(variation_types)]
    
    # Filter out incompatible variation combinations
    variation_subsets = filter_incompatible_variations(variation_subsets)
    
    # 2) Generate all entity swap combinations
    swap_dicts = generate_entity_swap_combinations(entity_list, original_entities)
    
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
            # Try to get entities from session state
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
        
        # Extract potential entities from the data for reference
        detected_entities = extract_entities_from_data(data)
        
        if detected_entities:
            st.write("### Detected entities in your data:")
            st.write(detected_entities)
        
        # Entity list input
        st.write("### Enter Entities for Variation:")
        
        # Initialize entity list in session state if not present
        if "entity_list" not in st.session_state:
            st.session_state["entity_list"] = []
        
        # Display current entity list
        if st.session_state["entity_list"]:
            st.write("#### Current Entity List:")
            for i, entity in enumerate(st.session_state["entity_list"]):
                st.write(f"{i+1}. {entity}")
        
        # Add new entity
        new_entity = st.text_input("New entity to add:", autocomplete="off")
        
        if st.button("Add Entity"):
            if new_entity and new_entity not in st.session_state["entity_list"]:
                st.session_state["entity_list"].append(new_entity)
                st.success(f"Added: '{new_entity}' to the entity list")
        
        # Option to clear the entity list
        if st.button("Clear Entity List"):
            st.session_state["entity_list"] = []
            st.success("Entity list cleared")
            
        # Manual override for original entities
        st.write("### Define Original Entities (that will be replaced):")
        orig_entity1 = st.text_input("Original Entity 1:", autocomplete="off")
        orig_entity2 = st.text_input("Original Entity 2:", autocomplete="off")
        
        original_entities = []
        if orig_entity1:
            original_entities.append(orig_entity1)
        if orig_entity2:
            original_entities.append(orig_entity2)
        
        # Store original entities in session state for reference in prompts
        if len(original_entities) == 2:
            st.session_state["original_entities"] = original_entities
        
        # Generate and view combinations
        if st.button("View Possible Variations"):
            if len(st.session_state["entity_list"]) > 0 and len(original_entities) < 2:
                st.error("Please specify both original entities to be replaced.")
            else:
                # Generate all combinations of variations and entity swaps
                all_variation_combinations = get_all_combinations(st.session_state["entity_list"], original_entities)
                
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
            if len(st.session_state["entity_list"]) > 0 and len(original_entities) < 2:
                st.error("Please specify both original entities to be replaced.")
            else:
                # Generate combinations if not already generated
                all_variation_combinations = get_all_combinations(st.session_state["entity_list"], original_entities)
                
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