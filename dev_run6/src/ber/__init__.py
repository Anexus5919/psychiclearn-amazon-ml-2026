"""Business Entity Resolution pipeline (team PsychicLearn, Amazon ML Challenge 2026).

Stages: prepare (normalise) -> block (candidate generation) -> features -> train (LightGBM,
out-of-fold validation, decision tuning) -> predict (write both submission files).
No external data or services are used; everything is learned from the provided files.
"""
