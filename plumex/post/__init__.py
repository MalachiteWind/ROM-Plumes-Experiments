"""Post analysis package.

This contains a variety of scripts to generate final figures from
trial data.  Modify the module constants for each script if new
experiments are run.  Run individual scripts to generate respective
figures/post analysis.  Run this module to generate all figures/post
analysis.
"""
from . import agu
from . import center
from . import edge_figs
from . import fig5
from . import points


def run():
    center.run()
    edge_figs.run()
    agu.run()
    fig5.run()
    points.run()


if __name__ == "__main__":
    run()

# __all__ = ["center", "edge_figs", "points", "fig5"]
