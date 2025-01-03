from flask import Flask, request, jsonify
from json import JSONEncoder
import numpy as np
from utils.crt import CrtEnc
from key_manager import KeyManager
from operations import SecureOps
import tenseal as ts

class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.ndarray, list)):
            return [float(x) for x in o]
        if isinstance(o, ts.CKKSTensor):
            ciphertexts = [ct.serialize().hex() for ct in obj]
            return ciphertexts
        if isinstance(o, bytes):
            return o.hex()
        return super().default(o)

app = Flask(__name__)
key_manager = KeyManager()
crt_enc = CrtEnc()
sec_ops = SecureOps(key_manager, crt_enc)


app.json_encoder = CustomJSONEncoder

@app.route('/init', methods=['POST'])
def initialize():
    context = key_manager.gen_keys()
    return jsonify({"status": "success", "public_keys": context})

@app.route('/process_ratings', methods=['POST'])
def process_ratings():
    try: 
        data = request.get_json()
        masked_data = data['masked_data']

        processed_data = sec_ops.process_rating(masked_data)
        return jsonify({
            "status": "success",
            "processed_data": processed_data
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
            }), 400

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
        "converged": conv
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
