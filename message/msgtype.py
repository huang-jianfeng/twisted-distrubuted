#更精确定义的一个枚举类，即value值不从默认的1开始
#枚举类
from enum import Enum, unique
@unique    #此装饰器可以帮助我们检查保证没有重复值
class MsgType(Enum):
    RegisterMsg = 0
    SendClientModelMsg = 1
    SendServerModelMsg = 2
    ReplyClientModelMsg = 3
    ReplyServerModelMsg = 4