import sys
sys.path.insert(0, r'c:\Users\arjun\welfog\welfog-ai\welfog-ai-agent\support')
from services.welfog_api import _split_product_query, fetch_products_from_api

queries = [
    'iphone and realme mobile covers',
    'bhai redmi ka cover aur iphone ka cover dikhana',
    'mobile cover aur charger',
]
for q in queries:
    print('QUERY:', q)
    parts = _split_product_query(q)
    print('SPLIT:', parts)
    for part in parts:
        res = fetch_products_from_api(part)
        print('  PART:', part, 'LEN:', len(res))
        if res:
            print('   FIRST:', res[0])
    print('')
