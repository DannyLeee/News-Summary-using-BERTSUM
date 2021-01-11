from flask import Flask, request, make_response
from get_summary import get_summary as __get_summary
from search import get_search_results
from multiprocess import Pool
import json

app = Flask(__name__)

PORT = 3000

@app.route("/", methods=["POST"])
def get_summary():
    #  source_text = request.form["sourceText"]
    source_text = request.get_json(force=True)["sourceText"]
    #  summary = f"This is a fake {mode} summary with {decoder}.\nThe long document:{source_text}"
    # summary = __get_summary(mode, decoder, source_text)
    with Pool(2) as pool:
        pool_output = pool.starmap(__get_summary, [("abs", "trans", source_text), ("ext", "trans", source_text)])
    abs_summary, ext_summary = pool_output[:2]
    search_results = get_search_results(abs_summary)
    summary = json.dumps({"abs": abs_summary, "ext": ext_summary[0], "highlighted": ext_summary[1], "searchResults": search_results}, ensure_ascii=False)
    print(summary)
    resp = make_response(summary)
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
pass

if __name__ == "__main__":
    app.run(host="140.118.127.72", port = PORT, debug=True)
pass
