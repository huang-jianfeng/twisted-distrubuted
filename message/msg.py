from message.msgtype import MsgType


class Message:
    def __init__(self,msgtype:MsgType) -> None:
        
        self.msgtype:MsgType = msgtype
