#!/usr/bin/env python3
# open_generalized_v2/__init__.py

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

__all__ = ["AdvancedGeneralizedOpenDoorEnv"]

def __getattr__(name):
    if name == "AdvancedGeneralizedOpenDoorEnv":
        try:
            from open_generalized_v2.env_v2 import AdvancedGeneralizedOpenDoorEnv
        except ModuleNotFoundError:
            from env_v2 import AdvancedGeneralizedOpenDoorEnv
        return AdvancedGeneralizedOpenDoorEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")