from flask import Flask, request, jsonify
import numpy as np
from utils.crt import CrtEnc
from key_manager import KeyManager
from operations import SecureOps

app = Flask(__name__)
key_manager = KeyManager()
crt_enc = CrtEnc()
sec_ops = SecureOps(crt_enc)

@app.route('/init', methods=['POST'])
def initialize():
    pub_keys = key_manager.gen_keys()
    return jsonify({"status": "success", "public_keys": pub_keys})

@app.route('/process_ratings', methods=['POST'])
def proccess_ratings():
    data = request.get_json()
    enc_ratings = data['encrypted_ratings']
    masks = data['masks']
    proccessed_data = sec_ops.process_rating(enc_ratings, masks)
    return jsonify({
        "status": "success",
        "proccessed_data": proccessed_data
        })

@app.route('/fixed_point_ops', methods=['POST'])
def fixed_ops():
    data = request.get_json()
    vecs = data['vectors']
    op = data['operation']
    res = sec_ops.fixed_point_compute(vecs, op)
    return jsonify({
        "status": "success",
        "result": res
        })

@app.route('/check_conv', methods=['POST'])
def check_conv():
    data = request.get_json()
    grads = data['gradients']
    thres = data['threshold']
    conv = sec_ops.check_convergence(grads, thres)
    return jsonify({
        "status": "success",
        "converaged": conv
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
