from typing import Tuple, Dict, List

from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import cgi
import json

from requests import models

from Optimizer import Optimizer, OptimizerDeterminist
from Models import EventModel, df_to_arr

class RequestHandler(BaseHTTPRequestHandler):
    
    def _set_headers(self):
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.end_headers()

    def _send_json_msg(self, msg: Dict):
        self._set_headers()
        self.wfile.write(json.dumps(msg).encode())

    def _read_json_msg(self) -> Dict:
        return json.loads(self.rfile.read(int(self.headers.get('content-length'))))

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'RpgTableSolver. POST request only.')

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        # refuse to receive non-json content
        if self.headers.get('content-type') != 'application/json':
            self.send_response(400)
            self.end_headers()
            return
            
        # read the message and convert it into a python dictionary
        message = self._read_json_msg()
        print(message)
        
        model = EventModel(max_parallel=4)

        # add a property to the object, just to mess with data
        model.from_array(
            message['slots'], 
            message['activities'], 
            message['preferences']
        )

        if model.clean_preferences():
            optimizer = OptimizerDeterminist()
            event = optimizer.optimize(model)
            
            # send the message back
            data = event.to_arr()
            data["analyse_happyness"] = df_to_arr(optimizer.compute_happyness_df(event))

            self._send_json_msg(data)

httpd = HTTPServer(('0.0.0.0', 8000), RequestHandler)
httpd.serve_forever()