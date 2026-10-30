# SPDX-License-Identifier: Apache-2.0
"""Opt-in displacement-bound compatibility for Mink 1.3.0.

Narrow adaptation of Mink's existing fix, not a new collision algorithm:
https://github.com/kevinzakka/mink/commit/1933012a7a8756e6c7732a7150155b0c7b46ec7b
Upstream Mink: Kevin Zakka and contributors, Apache-2.0.
Only the finite upper bound is corrected; installed packages stay unchanged.
"""
import hashlib
import inspect
from importlib.metadata import version

import numpy as np

FIX_COMMIT="1933012a7a8756e6c7732a7150155b0c7b46ec7b"


def install():
    import mink
    from mink.limits.limit import Constraint
    if version("mink") != "1.3.0":
        raise ValueError("This narrow backport is validated only for Mink 1.3.0")
    original=mink.CollisionAvoidanceLimit.compute_qp_inequalities
    if hasattr(original,"_displacement_backport"):
        return original._displacement_backport
    source=inspect.getsource(original)
    if "(gain * (dist - min_dist) / dt) + relaxation" not in source:
        raise ValueError("Unknown Mink implementation; refusing a possibly duplicate correction")
    def corrected(self,configuration,dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite integration step required")
        result=original(self,configuration,dt)
        if result.inactive:
            return result
        h=result.h.copy()
        finite=np.isfinite(h)
        # Original h = gain*(d-d_min)/dt + relaxation, but QP variable is Δq.
        # Preserve relaxation and inactive infinite rows, correcting only units.
        h[finite]=dt*(h[finite]-self.bound_relaxation)+self.bound_relaxation
        return Constraint(G=result.G,h=h)
    metadata={"profile":"upstream-displacement","mink_version":version("mink"),
              "upstream_fix_commit":FIX_COMMIT,
              "original_method_sha256":hashlib.sha256(source.encode()).hexdigest(),
              "changes":"Finite h: dt*(h-relaxation)+relaxation; same G, filtering, zero-gap and inactive rows",
              "penetration_recovery_forced":False,"installed_package_modified":False}
    corrected._displacement_backport=metadata
    mink.CollisionAvoidanceLimit.compute_qp_inequalities=corrected
    return metadata
