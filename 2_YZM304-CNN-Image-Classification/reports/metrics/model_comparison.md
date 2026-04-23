# Model Comparison

| Model | Kind | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Trainable Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lenet_like_cnn` | CNN | 0.9306 | 0.9372 | 0.9298 | 0.9299 | 5750 |
| `improved_batchnorm_dropout_cnn` | CNN | 0.9667 | 0.9685 | 0.9663 | 0.9661 | 5794 |
| `vgg_style_small_cnn` | CNN | 0.9583 | 0.9577 | 0.9578 | 0.9572 | 29722 |
| `hybrid_improved_cnn_features_svc` | CNN features + SVC | 0.9750 | 0.9751 | 0.9748 | 0.9747 | 0 |
