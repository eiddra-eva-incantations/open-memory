
import sys
import pprint
sys.path.append("/home/eiddra/mcp-servers/open-memory")
from mcp_server import mine_trajectories

print("Running mine_trajectories()...")
result = mine_trajectories()
pprint.pprint(result)
