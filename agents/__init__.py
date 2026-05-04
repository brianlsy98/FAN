from agents.fan import FANAgent
from agents.nbrac import NBRACAgent
from agents.nfql import NFQLAgent
from agents.faql import FAQLAgent

agents = dict(
    fan=FANAgent,
    faql=FAQLAgent,
    nfql=NFQLAgent,
    nbrac=NBRACAgent,
)
