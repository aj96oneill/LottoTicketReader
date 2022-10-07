from ticket import Ticket
from image_scanner import ImageScanner
from lotto_logic import MegaMillions

from flask import Flask, Response, request, send_file
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def test():
    return "Hello to flask server"

@app.route("/check_ticket", methods=["GET"])
def check():
    lotto = request.args.get("lotto")
    image = request.files['ticket']
    image.save(os.path.join(app.root_path, image.filename))

    scanned = ImageScanner().scan(os.path.join(app.root_path, image.filename))
    ticket = Ticket(scanned, lotto)
    if not ticket.process_ticket(): print("error")
    MegaMillions(ticket).check_nums()
    
    filename = image.filename.split(".")[0]+"_results.png"
    return send_file(filename, mimetype='image/png')


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)