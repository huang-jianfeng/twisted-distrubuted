import pickle
import sys 
from pathlib import Path

import torch

from utils.utils import get_best_device
# sys.path.append("..

PROJECT_DIR = Path(__file__).parent.parent.absolute()

sys.path.append(PROJECT_DIR.as_posix())
# sys.path.append(PROJECT_DIR.joinpath("src").as_posix())

from argparse import ArgumentParser, Namespace
from ast import Dict, List
from twisted.internet.protocol import Protocol, connectionDone,Factory
from twisted.python import failure
from twisted.internet import reactor
from twisted.protocols.basic import Int32StringReceiver
from message.msg import Message
import logging

from message.msgtype import MsgType
from server.clientstatus import ClientStatus


def get_fedavg_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lenet5",
        choices=["lenet5", "2nn", "avgcnn", "mobile", "res18", "alex", "sqz"],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "synthetic",
            "femnist",
            "emnist",
            "fmnist",
            "celeba",
            "medmnistS",
            "medmnistA",
            "medmnistC",
            "covid19",
            "svhn",
            "usps",
            "tiny_imagenet",
            "cinic10",
            "domain",
        ],
        default="cifar10",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-jr", "--join_ratio", type=float, default=0.1)
    parser.add_argument("-ge", "--global_epoch", type=int, default=100)
    parser.add_argument("-le", "--local_epoch", type=int, default=1)
    parser.add_argument("-fe", "--finetune_epoch", type=int, default=0)
    parser.add_argument("-tg", "--test_gap", type=int, default=100)
    parser.add_argument("-ee", "--eval_test", type=int, default=1)
    parser.add_argument("-er", "--eval_train", type=int, default=0)
    parser.add_argument("-lr", "--local_lr", type=float, default=1e-2)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-vg", "--verbose_gap", type=int, default=100000)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-v", "--visible", type=int, default=0)
    parser.add_argument("--global_testset", type=int, default=0)
    parser.add_argument("--straggler_ratio", type=float, default=0)
    parser.add_argument("--straggler_min_local_epoch", type=int, default=1)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--save_log", type=int, default=1)
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--save_fig", type=int, default=1)
    parser.add_argument("--save_metrics", type=int, default=1)
    
    parser.add_argument("--client_num",type=int,required=True)
    return parser


class Fedavg:

    def __init__(self) -> None:
        pass


class Server(Int32StringReceiver):

    def __init__(self, model, args: Namespace, logger: logging.Logger):
        self.logger = logger
        self.model:torch.nn.Module = model
        self.global_params_dict = self.model.state_dict()
        self.args = args
        self.clientNum = args.client_num
        self.clinetModels:List[torch.nn.Module] = []
        self.deltaCache = []
        self.weightCache = []
        self.numSamplesCache = []

        self.selectedClients:List = []
        self.rounds = 0

        self.device = get_best_device(self.args.use_cuda)
        
        self.clientStatus:Dict[int,ClientStatus] = {}
        for i in range(self.clientNum):
            self.clientStatus[i] = ClientStatus(i,False)

    def stringReceived(self, payload):
        msg = pickle.loads(payload)
        self.process(msg)

    def connectionLost(self, reason: failure.Failure = ...):
       self.logger.info("test connectionLost")
    
    def connectionMade(self):
        self.logger.info("connectionMade")

    def process(self,msg:Message):
        if msg.msgtype == MsgType.RegisterMsg:
            self.registerClient(msg.getClientId())
            self.logger.info("client:{} register.".format(msg.getClientId()))

        elif msg.msgtype == MsgType.SendClientModelMsg:
            if msg.rounds == self.rounds:
                self.deltaCache.append(msg.modelDict)
                self.weightCache.append(msg.weightCache)
                self.numSamplesCache.append(msg.numSamples)
                if len(self.deltaCache) == len(self.selectedClients):
                    self.aggregate(self.deltaCache,self.weightCache)
            


    @torch.no_grad()
    def aggregate(
        self,
        delta_cache: List[List[torch.Tensor]],
        weight_cache: List[int],
        return_diff=True,
    ):
        """
        This function is for aggregating recevied model parameters from selected clients.
        The method of aggregation is weighted averaging by default.

        Args:
            delta_cache (List[List[torch.Tensor]]): `delta` means the difference between client model parameters that before and after local training.

            weight_cache (List[int]): Weight for each `delta` (client dataset size by default).

            return_diff (bool): Differnt value brings different operations. Default to True.
        """
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        if return_diff:
            delta_list = [list(delta.values()) for delta in delta_cache]
            aggregated_delta = [
                torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                for diff in zip(*delta_list)
            ]

            for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
                param.data -= diff
        else:
            for old_param, zipped_new_param in zip(
                self.global_params_dict.values(), zip(*delta_cache)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )
        self.model.load_state_dict(self.global_params_dict, strict=False)
            
        
    
    def registerClient(self,clientId:int):
        self.clientStatus[clientId].connecting()


class ServerFactory(Factory):

    def __init__(self,model,args,logger):

        self.logger = logger
        self.args = args
        self.model = model


    def buildProtocol(self, addr):
        return Server(self.model,self.args,self.logger)

if __name__ == '__main__':

    args = get_fedavg_argparser().parse_args()

    logger = logging.getLogger('server-logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    factory = ServerFactory(None,args,logger)
    reactor.listenTCP(8007,factory)
    reactor.run()