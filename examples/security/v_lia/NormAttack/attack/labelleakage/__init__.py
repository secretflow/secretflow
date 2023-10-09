"""Subpackage for label leakage attack, which infere the private label information of the training dataset.
"""

from .normattack import (NormAttackSplitNNManager,  # noqa: F401
                         NormAttackSplitNNManager_sf,
                         attach_normattack_to_splitnn,
                         attach_normattack_to_splitnn_sf)
