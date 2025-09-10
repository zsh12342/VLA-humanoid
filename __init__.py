"""
VLA Robot Learning Framework

A minimal runnable system for collecting, training, and executing 
"instruction-perception-action" triplet data in MuJoCo simulation 
with support for future real robot integration.
"""

__version__ = "0.1.0"
__author__ = "VLA Team"
__email__ = "vla@example.com"

from .config import config
from .modules.instruction import InstructionModule
from .modules.perception import PerceptionModule
from .modules.skill import Skill, WalkSkill, TurnSkill, CustomSkill, SkillManager
from .modules.collector import Collector
from .modules.algorithm import System2Planner, System1Executor, Trainer, TrajectoryDataset

__all__ = [
    # Configuration
    "config",
    
    # Core modules
    "InstructionModule",
    "PerceptionModule", 
    "Skill",
    "WalkSkill",
    "TurnSkill", 
    "CustomSkill",
    "SkillManager",
    "Collector",
    "System2Planner",
    "System1Executor", 
    "Trainer",
    "TrajectoryDataset",
]

# Framework version info
def get_version():
    """Get framework version"""
    return __version__

def get_info():
    """Get framework information"""
    return {
        "name": "VLA Robot Learning Framework",
        "version": __version__,
        "description": "Minimal runnable system for robot learning",
        "author": __author__,
        "email": __email__,
        "modules": __all__
    }