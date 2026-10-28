# maybe separate scripts per model?

# at least 3 models
# 1) CNN
# 2) Vision Transformer (ViT)
# 3) Mobile ViT

"""
imagined manual data splitting:

train/test/validation = 80/10/10
MTG: 100 
PKM: 100
YGO: 100
random selection w/out replacement
train: 80% of MTG, 80% of PKM, 80% of YGO
    80 MTG, 80 PKM, 80 YGO = 240 total, equal classes
test: 50% of remaining MTG, 50% of remaining PKM, 50% of remaining YGO
    10 MTG, 10 PKM, 10 YGO = 30 total, equal classes
validation: remaining MTG, remaining PKM, remaining YGO
    10 MTG, 10 PKM, 10 YGO = 30 total, equal classes

train folder:
    parquet-001
    parquet-002
    parquet-003
    ...
    parquet-009
test folder:
    parquet-001
    parquet-002
validate folder:
    parquet-001
    parquet-002
"""