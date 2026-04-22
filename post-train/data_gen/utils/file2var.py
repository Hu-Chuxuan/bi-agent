import re

SQLITE_KEYWORDS = {"index", "order", "select", "from", "table"}  # add more as needed

def sanitize_name(name: str) -> str:
    name = name.replace("'", "").replace('"', '')
    name = re.sub(r'[^A-Za-z0-9]', '_', name)
    if re.match(r'^\d', name):
        name = '_' + name
    name = name.lower()
    if name in SQLITE_KEYWORDS:
        name = name + "_col"  # or prepend _
    return name
