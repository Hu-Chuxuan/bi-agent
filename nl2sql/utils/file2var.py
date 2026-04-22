import re

def sanitize_name(name: str) -> str:

    # Remove single and double quotes
    name = name.replace("'", "").replace('"', '')

    # Replace all other non-alphanumeric characters with underscores
    name = re.sub(r'[^A-Za-z0-9]', '_', name)

    # Prepend underscore if the name starts with a digit
    if re.match(r'^\d', name):
        name = '_' + name

    return name
