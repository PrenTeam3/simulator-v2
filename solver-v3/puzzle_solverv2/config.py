"""Global solver settings — single source of truth for all tuneable parameters."""

# Allowed ±error (mm) when checking whether a frame side is fully covered.
# Used by both the Step 6 search (candidate filtering) and the Step 7 C2 constraint.
SIDE_TOLERANCE_MM: float = 15.0

# Maximum allowed overlap area (mm²) between any two placed pieces.
# Overlaps beyond this threshold indicate a geometrically inconsistent placement.
MAX_OVERLAP_MM2: float = 500.0
