class ClientStatus:

    def __init__(self,clientId:int,connnected:bool) -> None:

        self.clientId = clientId
        self.connected = connnected

  
    def connecting(self)->None:
        self.connected = True
        

    def loseConnected(self)->None:
        self.connected = False

        
        