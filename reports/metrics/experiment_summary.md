# Experiment Summary

Best experiment: `manual_raw_baseline`

| Experiment | Model | Standardized | Hidden Layers | Steps | Val Acc | Test Acc | Test F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| manual_raw_baseline | manual | False | 6 | 500 | 0.9727 | 0.9635 | 0.9576 |
| manual_standardized_baseline | manual | True | 6 | 1000 | 0.9636 | 0.9745 | 0.9719 |
| manual_regularized_deeper | manual | True | 10-6 | 1000 | 0.9591 | 0.9708 | 0.9680 |
| sklearn_standardized_baseline | sklearn | True | 6 | 1000 | 0.9591 | 0.9708 | 0.9677 |
