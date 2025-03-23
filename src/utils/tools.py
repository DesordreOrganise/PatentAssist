import re

def get_base_id(id: str) -> str:
    match = re.match(r'\b(?:Article|Art\.|Rule|R\.)\s\d+[a-zA-Z]?\b', id)
    return match.group()