"""One collector per environment family, with exact preset dispatch."""

from .base import BaseCollector, CollectionResult
from .ball_catcher import BallCatcherCollector
from .hand_pick_up import HandPickUpCollector
from .locomotion import LocomotionCollector
from .lq import LQCollector
from .microgrid import MicrogridCollector
from .mobile import MobileCollector
from .nerograsp import NeroGraspCollector
from .pokemon import PokemonCollector
from .racing import RacingCollector
from .rotate_in_hand import RotateInHandCollector
from .soft_walker import SoftWalkerCollector
from .solar_cleaner import SolarCleanerCollector
from .truck_unload import TruckUnloadCollector
from .warehouse import WarehouseCollector
from .storage import CorpusReader

COLLECTOR_CLASSES = (BallCatcherCollector, HandPickUpCollector, LocomotionCollector, LQCollector,
                     MicrogridCollector, MobileCollector, NeroGraspCollector, PokemonCollector,
                     RacingCollector, RotateInHandCollector, SoftWalkerCollector,
                     SolarCleanerCollector, TruckUnloadCollector, WarehouseCollector)
COLLECTORS = {preset: cls for cls in COLLECTOR_CLASSES for preset in cls.presets}


def make_collector(env_id, **kwargs):
    try:
        cls = COLLECTORS[env_id]
    except KeyError as error:
        raise ValueError(f"Unsupported LOKI preset {env_id!r}; choose from {sorted(COLLECTORS)}") from error
    return cls(env_id, **kwargs)


__all__ = ["BaseCollector", "CollectionResult", "CorpusReader", "COLLECTORS",
           "make_collector", *(cls.__name__ for cls in COLLECTOR_CLASSES)]
