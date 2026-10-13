import os
import streamlit as st
import re
import json

import shared_config


# --- Chat History Functions ---
def get_saved_chats():
    """Returns a list of filenames of saved chats, sorted by modification time (newest first)."""
    try:
        # 1. Get all .json files from the directory
        files = [f for f in os.listdir(shared_config.CHAT_HISTORY_DIR) if f.endswith(".json")]
        
        # 2. Create a list of full filepaths
        filepaths = [os.path.join(shared_config.CHAT_HISTORY_DIR, f) for f in files]
        
        # 3. Sort the list of filepaths based on their modification time (newest first)
        filepaths_sorted = sorted(filepaths, key=os.path.getmtime, reverse=True)
        
        # 4. Extract just the filenames (not the full path) from the sorted list
        filenames_sorted = [os.path.basename(f) for f in filepaths_sorted]
        
        return filenames_sorted
        
    except FileNotFoundError:
        st.sidebar.warning(f"Chat history folder not found at: {shared_config.CHAT_HISTORY_DIR}")
        return []
    except Exception as e:
        st.sidebar.error(f"Error reading chat history: {e}")
        return []

def sanitize_filename(name):
    """Removes invalid characters for filenames and adds .json extension."""
    # Strip .json suffix if it exists, to prevent 'my_chat_json.json'
    if name.endswith(".json"):
        name = name[:-5]
    # Remove characters that are not alphanumeric, underscore, or hyphen
    sanitized = re.sub(r'[^\w\-]+', '_', name)
    # Truncate to a reasonable length
    sanitized = sanitized[:50]
    # Remove leading/trailing underscores/hyphens
    sanitized = sanitized.strip('_').strip('-')
    # Ensure filename is not empty, default to 'chat' if it is
    if not sanitized:
        sanitized = "chat"
    return f"{sanitized}.json"

def desanitize_filename(filename):
    """Converts a filename like 'My_Chat.json' back to a friendly name 'My Chat'."""
    if filename.endswith(".json"):
        name = filename[:-5]
    else:
        name = filename
    
    # Replace underscores with spaces
    friendly_name = name.replace('_', ' ')
    return friendly_name

def save_chat_history(messages, filename):
    """Saves the current chat messages to a JSON file, 'filename' is expected to be pre-sanitized."""
    filepath = os.path.join(shared_config.CHAT_HISTORY_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
        st.sidebar.success(f"Chat saved as {filename}")
        return filename # Return filename to update dropdown selection
    except Exception as e:
        st.sidebar.error(f"Error saving chat: {e}")
        return None

def load_chat_history(filename):
    """Loads chat messages from a JSON file."""
    filepath = os.path.join(shared_config.CHAT_HISTORY_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        # Add basic validation
        if isinstance(messages, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
             # Ensure sources are handled correctly if they exist (basic check)
            for msg in messages:
                if msg.get("role") == "assistant" and "sources" in msg:
                    if not isinstance(msg["sources"], list):
                        msg.pop("sources") # Remove invalid sources
                    else:
                         # Ensure each source is a dictionary with expected keys
                        msg["sources"] = [
                            s for s in msg["sources"]
                            if isinstance(s, dict) and "page_content" in s and "metadata" in s
                        ]
            return messages
        else:
            st.sidebar.error("Invalid chat file format.")
            return []
    except FileNotFoundError:
        st.sidebar.error("Chat file not found.")
        return []
    except json.JSONDecodeError:
        st.sidebar.error("Error decoding chat file (invalid JSON).")
        return []
    except Exception as e:
        st.sidebar.error(f"Error loading chat: {e}")
        return []
    
def delete_chat_history(filename):
    """Deletes a specific chat history JSON file."""
    if not filename:
        return False
    
    filepath = os.path.join(shared_config.CHAT_HISTORY_DIR, filename)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted old chat file: {filename}")
            return True
        else:
            print(f"File to delete not found: {filename}")
            return False # Not an error, just didn't exist
    except Exception as e:
        st.sidebar.error(f"Error deleting old chat file ({filename}): {e}")
        return False