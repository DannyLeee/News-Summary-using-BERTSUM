from flask import Flask, request, make_response
from get_summary import get_summary as __get_summary

app = Flask(__name__)

PORT = 3000

@app.route("/<string:mode>/<string:decoder>", methods=["POST"])
def get_summary(mode, decoder):
    #  source_text = request.form["sourceText"]
    source_text = request.get_json(force=True)["sourceText"]
    #  summary = f"This is a fake {mode} summary with {decoder}.\nThe long document:{source_text}"
    summary = __get_summary(mode, decoder, source_text)
    resp = make_response({"summary": summary})
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
pass

if __name__ == "__main__":
    app.run(host="140.118.127.72", port = PORT, debug=True)
pass
