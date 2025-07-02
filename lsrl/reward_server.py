import json, os, shutil, re, random, io, time, random, re, math
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from bottle import request
import bottle
import torch
import torch.nn as nn

from .utils import json_to_bytes_list, bytes_list_to_json

class RewardServer:
    def __init__(self, model_path, host='0.0.0.0', port=59878):       
        self.app = bottle.Bottle()
        self.host = host
        self.port = port
        self.init(model_path)

    def get_reward(self, data):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def init(self, model_path):
        raise NotImplementedError("This method should be implemented in subclasses.")
            
    def run_server(self): 
        @self.app.route('/get_reward', method='POST')
        def get_reward():
            dd = request.body.read()
            data = bytes_list_to_json(dd)
            return self.get_reward(data)

        bottle.run(self.app, host=self.host, port=self.port, server='tornado')

    def start(self):
        self.run_server()
    