from twisted.internet.protocol import Protocol, connectionDone,Factory
from twisted.python import failure
from twisted.internet import reactor
from twisted.protocols.basic import Int32StringReceiver

import logging

class Echo(Int32StringReceiver):


    def stringReceived(self, string):

        print(string)


    def connectionLost(self, reason: failure.Failure = ...):

       logging.info("test connectionLost")
    
    def connectionMade(self):
        logging.info("connectionMade")

if __name__ == '__main__':
    factory = Factory()
    factory.protocol = Echo

    reactor.listenTCP(8007,factory)
    reactor.run()