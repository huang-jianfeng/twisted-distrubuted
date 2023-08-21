from twisted.internet.protocol import Protocol,ClientCreator,ClientFactory
from twisted.internet import reactor
from twisted.protocols.basic import Int32StringReceiver
from sys import stdout
import logging
class Echo(Int32StringReceiver):

    def stringReceived(self, string):
        print("from server:".format(string))


    def connectionMade(self):
        logging.info("connection>")
        self.sendString(b"hello from client.")
        

class EchoClientFactory(ClientFactory):
    def startedConnecting(self, connector):
        print ('Started to connect.')
    
    def buildProtocol(self, addr):
        print ('Connected.')
        return Echo()
    
    def clientConnectionLost(self, connector, reason):
        print ('Lost connection.  Reason:', reason)
    
    def clientConnectionFailed(self, connector, reason):
        print('Connection failed. Reason:', reason)

if __name__ =='__main__':
    reactor.connectTCP('127.0.0.1',8007,EchoClientFactory())

    reactor.run()