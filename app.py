from flask import Flask, request, make_response
from get_summary import get_summary as __get_summary
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
    summary = json.dumps({"abs":pool_output[0], "ext":pool_output[1]}, ensure_ascii=False)    
    resp = make_response(summary)
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
pass

if __name__ == "__main__":
    app.run(host="140.118.127.72", port = PORT, debug=True)
pass
