import os
import json
import sqlite3
import shutil
import time

# Create a mock chat structure
mock_dir = os.path.expanduser("~/.gemini/tmp/eiddra/chats")
os.makedirs(mock_dir, exist_ok=True)

mock_chat = {
    "id": "1234-mock-chat",
    "messages": [
        {"type": "user", "content": [{"text": "Can you list my home directory and read config.json?"}]},
        {"type": "gemini", "content": "I'll do that for you.", "toolCalls": [{"name": "run_command", "args": {"command": "ls -la ~"}}]},
        {"type": "user", "content": [{"text": "Tool result: config.json file.txt"}]},
        {"type": "gemini", "content": "I see the config.json. Let me read it.", "toolCalls": [{"name": "view_file", "args": {"path": "~/config.json"}}]},
        {"type": "user", "content": [{"text": "Tool result: {\"setting\": true}"}]},
        {"type": "gemini", "content": "The setting is true."}
    ]
}

test_file = os.path.join(mock_dir, "test_chat_for_mining.json")
with open(test_file, "w") as f:
    json.dump(mock_chat, f)

print(f"Created mock chat file: {test_file}")

# To test this, you'd usually run mcp_server.mine_trajectories(). Let's run it via a small wrapper.
wrapper = """
import sys
import pprint
sys.path.append("/home/eiddra/mcp-servers/open-memory")
from mcp_server import mine_trajectories

print("Running mine_trajectories()...")
result = mine_trajectories()
pprint.pprint(result)
"""

with open("test_runner.py", "w") as f:
    f.write(wrapper)
    
print("Created test runner. Use python3 test_runner.py to execute the tool.")
