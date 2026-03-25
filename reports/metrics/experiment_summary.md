# Experiment Summary

Best experiment: `manual_raw_baseline`

| Experiment | Backend | Train Fraction | Standardized | Hidden Layers | Steps | Params | Val Acc | Test Acc | Test F1 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| manual_raw_baseline | manual | 1.00 | False | 6 | 500 | 37 | 0.9727 | 0.9635 | 0.9576 |
| manual_standardized_baseline | manual | 1.00 | True | 6 | 1000 | 37 | 0.9636 | 0.9745 | 0.9719 |
| sklearn_standardized_baseline | sklearn | 1.00 | True | 6 | 1000 | 37 | 0.9636 | 0.9745 | 0.9719 |
| pytorch_standardized_baseline | pytorch | 1.00 | True | 6 | 1000 | 37 | 0.9636 | 0.9745 | 0.9719 |
| manual_regularized_deeper_data50 | manual | 0.50 | True | 10-6 | 1000 | 123 | 0.9591 | 0.9745 | 0.9719 |
| manual_regularized_deeper_data75 | manual | 0.75 | True | 10-6 | 1000 | 123 | 0.9591 | 0.9708 | 0.9680 |
| manual_regularized_deeper_data100 | manual | 1.00 | True | 10-6 | 1000 | 123 | 0.9591 | 0.9708 | 0.9680 |
| sklearn_regularized_deeper | sklearn | 1.00 | True | 10-6 | 1000 | 123 | 0.9591 | 0.9708 | 0.9680 |
| pytorch_regularized_deeper | pytorch | 1.00 | True | 10-6 | 1000 | 123 | 0.9591 | 0.9708 | 0.9680 |
