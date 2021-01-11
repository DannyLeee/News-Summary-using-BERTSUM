import env
import json
import requests

KEY = env.KEY
CX = "8908903ee441a922b" # Google custom search engine id
BASE_URL = f"https://www.googleapis.com/customsearch/v1?key={KEY}&cx={CX}&q="

def get_search_results(query):
    res = requests.get(BASE_URL + query)

    if not "items" in res.json().keys():
        return []

    else:
        items = res.json()["items"]

        result_items = []
        for i in items:
            title = i["htmlTitle"]
            link = i["formattedUrl"]
            snippet = i["htmlSnippet"]

            result_item = {}
            for var in ["title", "link", "snippet"]:
                result_item[var] = eval(var)

            result_items.append(result_item)

        return result_items