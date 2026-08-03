import sys
import datetime

# Try to get value from argument, otherwise fallback to built version
if len(sys.argv) < 2:
    from lightworks import __version__ as ver
    current_ver = str(ver)
else:
    raw_ver = sys.argv[1]
    current_ver = raw_ver.lstrip('v')
    
current_date = datetime.datetime.now(tz=datetime.UTC)

version_str = f"version: v{current_ver}\n"
date_str = (
    f"date-released: '{current_date.year:04d}-{current_date.month:02d}-"
    f"{current_date.day:02d}'\n"
)

with open("CITATION.cff", encoding="utf-8") as f:
    all_lines = f.readlines()

with open("CITATION.cff", "w", encoding="utf-8") as f:
    for line in all_lines:
        if line.startswith("date-released:"):
            f.write(date_str)
        elif line.startswith("version:"):
            f.write(version_str)
        else:
            f.write(line)
