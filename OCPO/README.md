# OCPO core code

`update.py` contains the overlap-aware factor construction, factor log
probability, and team-advantage clipped actor loss used by OCPO. The runnable
training and evaluation entry point is `major/cross_scene/run.py`; from the
repository root, use `python -B review.py train --method ocpo --seed 123`.

`MAPPO/` holds the comparison method's shared network definition and archived
vehicle scene files. The OCPO update is implemented here and selected by
`--method ocpo` in the shared experiment runner.
