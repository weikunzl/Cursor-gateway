"""
Extract protobuf definitions from Cursor's workbench JS bundle.
"""
import re

with open("/Applications/Cursor.app/Contents/Resources/app/out/vs/workbench/workbench.desktop.main.js", "r") as f:
    data = f.read()

# Find all class definitions with aiserver.v1 typeNames
pattern = r'typeName="(aiserver\.v1\.[^"]+)"\}static\{this\.fields=_.util\.newFieldList\(\(\)=\>\[(.*?)\]\)\}'
matches = re.findall(pattern, data)

for typename, fields_str in matches:
    if "StreamUnifiedChat" in typename or "ConversationMessage" in typename or "ModelDetails" in typename:
        print(f"\n=== {typename} ===")
        # Parse individual fields - simplified
        field_pattern = r'\{no:(\d+),name:"([^"]+)",kind:"([^"]+)"'
        for m in re.finditer(field_pattern, fields_str):
            print(f"  {m.group(1)}: {m.group(2)} ({m.group(3)})")
