import json
from mcp_server import dream, generate_morning_report

print("Initiating manual flagship Dream cycle...")
result = dream()
print(result)

print("
Generating Morning Cognitive Report preview...")
report = generate_morning_report()
print(json.dumps(report, indent=2))
