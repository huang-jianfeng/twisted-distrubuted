from message.msg import Message
from message.msgtype import MsgType


class SendClientModelMsg(Message):

    def __init__(self,clientId:int,modelDict,rounds:int,weight,delta,numSamples:int):
        super().__init__(MsgType.SendClientModelMsg)
        self.clientId = clientId
        self.modelDict = modelDict
        self.rounds:int = rounds
        self.weight:float = weight
        self.delta = delta
        self.numSamples:int = numSamples