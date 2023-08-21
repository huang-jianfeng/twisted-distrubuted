import sys
from pathlib import Path
from typing import Dict

PROJECT_DIR = Path(__file__).parent.parent.absolute()

sys.path.append(PROJECT_DIR.as_posix())
from argparse import Namespace
import pickle
from twisted.internet.protocol import Protocol,ClientCreator,ClientFactory
import torch
from twisted.internet import reactor
from twisted.protocols.basic import Int32StringReceiver
from sys import stdout
import logging

from message.register import RegisterMsg
from server.fedavg import get_fedavg_argparser
import torch.nn as nn

class FedAvgClient:
    clientId:int
    def __init__(self,model:nn.Module,clientId:int,logger:logging.Logger) -> None:
        self.model:nn.Module = model
        self.clientId:int = clientId
        self.logger:logging.Logger = logger

 
    def getClientId(self):
        return self.clientId
    
    def getParameters(self)->Dict:
        return self.model.state_dict

    def setParameters(self,parameter)->None:
        self.model.load_state_dict(parameter,strict=False)
        
class CommunicateClient(Int32StringReceiver):
    fedClient:FedAvgClient
    def __init__(self,fedClient:FedAvgClient,logger:logging.Logger) -> None:
        self.fedClient = fedClient
        self.logger:logging.Logger = logger


    def stringReceived(self, string):
        self.logger.info("from server:".format(string))
        
    
    def connectionMade(self):
        logging.info("connection>")
        msg = RegisterMsg(self.fedClient.getClientId())
        payload = pickle.dumps(msg) 
        self.sendString(payload)
        

class CommunicateClientFactory(ClientFactory):
    def __init__(self,args:Namespace,clientId:int,logger:logging.Logger) -> None:

        self.clientId = clientId
        self.logger = logger
        self.args = args

    def startedConnecting(self, connector):
        print ('Started to connect.')
    
    def buildProtocol(self, addr):
        print ('Connected.')
        fedAvgClient=FedAvgClient(None,self.clientId,self.logger)
        t = fedAvgClient.getClientId()
        return CommunicateClient(fedAvgClient,self.logger)
    
    def clientConnectionLost(self, connector, reason):
        self.logger.info('Lost connection.  Reason:', reason)
    
    def clientConnectionFailed(self, connector, reason):
        self.logger.info('Connection failed. Reason:', reason)

if __name__ =='__main__':
    server='127.0.0.1'
    port = 8007

    logger = logging.getLogger('server-logger')
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    args = get_fedavg_argparser().parse_args()
    reactor.connectTCP(server,port,CommunicateClientFactory(args,1,logger))

    reactor.run()