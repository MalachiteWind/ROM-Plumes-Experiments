"""Post analysis package.

This contains a variety of scripts to generate final figures from
trial data.  Modify the module constants for each script if new
experiments are run.  Run individual scripts to generate respective
figures/post analysis.  Run this module to generate all figures/post
analysis.
"""
from . import edge_figs

__all__ = ["edge_figs"]
