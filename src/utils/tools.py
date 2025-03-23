import re

def get_base_id(id: str) -> str:
    match = re.match(r'\b(?:Article|Art\.|Rule|R\.)\s\d+[a-zA-Z]?\b', id)
    return match.group()


def extract_articles(text: str) -> list:
    # Définir le pattern de recherche
    article_pattern = r'Art\. \d+[\w\s\(\)]+(?:\sEPC|PCT|Paris Convention|OJ|Guidelines|Rfees)?'
    article_pattern = r'Art\. \d+[\w\s\(\)]+(?=\s|$)'
    article_pattern = r'Art\. \d+[\w\s\(\)]+'
    article_pattern = re.compile(r'(?<![a-zA-Z0-9])Art(?:icle|icles|.)?\s*\d+[\w\s\(\),;.]*?(?:\sEPC|EPC|PCT|Paris Convention|OJ|Guidelines|Rfees)?', re.IGNORECASE)
    

    # replace \n | \r | \t | \s+ with space
    text = re.sub(r'[\n\r\t]', ' ', text)
    # Extraire les articles et les règles
    articles = re.findall(article_pattern, text)

    process = []
    for article in articles:
        if article.strip().endswith('EPC'):
            article = article.strip()[:-3]

        process.append(article.strip())

    # Supprimer les doublons
    articles = list(set(process))

    articles = sorted(articles, key=lambda x: int(re.findall(r'\d+', x)[0]))
    for article in articles:
        print(f'"{article}"')
        print()
        
    return articles

def extract_rules(text: str) -> list:
    # Définir le pattern de recherche
    rule_pattern = r'R\. \d+[\w\s\(\)]+'
    rule_pattern = re.compile(r'(?<![a-zA-Z0-9])R(?:.|ule|ules|)\s*\d+[\w\s\(\),;.]*?(?:\sEPC|EPC|PCT|Paris Convention|OJ|Guidelines|Rfees)?', re.IGNORECASE)
    # rule_pattern = r'Rule \d+[\w\s\(\)]+'

    # replace \n | \r | \t | \s+ with space
    text = re.sub(r'[\n\r\t]', ' ', text)

    # Extraire les articles et les règles
    rules = re.findall(rule_pattern, text)

    process = []
    for rule in rules:
        if rule.endswith('EPC'):
            rule = rule[:-3]
        process.append(rule.strip())


    # Supprimer les doublons
    rules = list(set(process))

    rules = sorted(rules, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for rule in rules:
        print(rule)
        
    return rules