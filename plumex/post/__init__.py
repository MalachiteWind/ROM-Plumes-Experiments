"""Post analysis package.

This contains a variety of scripts to generate final figures from
trial data.  Modify the module constants for each script if new
experiments are run.  Run individual scripts to generate respective
figures/post analysis.  Run this module to generate all figures/post
analysis.
"""
from . import center
from . import edge_figs
from . import fig5
from . import points

__all__ = ["center", "edge_figs", "points", "fig5"]
