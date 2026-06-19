#!/usr/bin/env python3
# open_generalized_v2/__init__.py
#
# Generalizzazione della Generalizzazione del task di APERTURA della porta.
# Pacchetto SPECULARE a close_generalized_v2 (chiusura), per il SOLO curriculum 1
# (posa variabile, soglie adattive, fisica randomizzata).
#
# NB: l'import di AdvancedGeneralizedOpenDoorEnv è LAZY: importare il package non deve
# forzare il caricamento di gymnasium/robosuite (così i test offline e la sola lettura
# del config/fsm funzionano senza quelle dipendenze pesanti).

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

__all__ = ["AdvancedGeneralizedOpenDoorEnv"]


def __getattr__(name):
    # PEP 562: import pigro a livello di modulo.
    if name == "AdvancedGeneralizedOpenDoorEnv":
        try:
            from open_generalized_v2.env_v2 import AdvancedGeneralizedOpenDoorEnv
        except ModuleNotFoundError:
            from env_v2 import AdvancedGeneralizedOpenDoorEnv
        return AdvancedGeneralizedOpenDoorEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
