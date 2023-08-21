from message.msg import Message

from message.msgtype import MsgType


class RegisterMsg(Message):
    
    def __init__(self,clientId:int) -> None:
        
        super().__init__(MsgType.RegisterMsg)
        self.msgtype = MsgType.RegisterMsg
        self.clientId = clientId


    def getClientId(self)->int:
        return self.clientId