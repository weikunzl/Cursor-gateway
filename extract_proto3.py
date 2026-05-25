"""
Extract specific protobuf definitions.
"""
import re

with open("/Applications/Cursor.app/Contents/Resources/app/out/vs/workbench/workbench.desktop.main.js", "r") as f:
    data = f.read()

pattern = r'typeName="(aiserver\.v1\.[^"]+)"\}static\{this\.fields=_.util\.newFieldList\(\(\)=\>\[(.*?)\]\)\}'
matches = re.findall(pattern, data)

targets = ["StreamUnifiedChatRequestWithToolsIdempotent", "StreamUnifiedChatResponseWithToolsIdempotent", "BidiRequestId"]
for typename, fields_str in matches:
    short = typename.replace("aiserver.v1.", "")
    if short in targets:
        print(f"\n=== {typename} ===")
        field_pattern = r'\{no:(\d+),name:"([^"]+)",kind:"([^"]+)"'
        for m in re.finditer(field_pattern, fields_str):
            print(f"  {m.group(1)}: {m.group(2)} ({m.group(3)})")
