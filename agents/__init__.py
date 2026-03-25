from agents.fan import FANAgent
from agents.nbrac import NBRACAgent
from agents.nfql import NFQLAgent
from agents.faql import FAQLAgent

from agents.fql import FQLAgent
from agents.value_flows import ValueFlowsAgent

agents = dict(
    fan=FANAgent,
    faql=FAQLAgent,
    nfql=NFQLAgent,
    nbrac=NBRACAgent,

    fql=FQLAgent,
    value_flows=ValueFlowsAgent,
)