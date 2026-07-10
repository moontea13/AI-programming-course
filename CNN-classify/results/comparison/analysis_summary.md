# Result Comparison Summary

| Model | Params | Best Epoch | Best Val Acc | Best Test Acc | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| SimpleCNN | 0.62M | 9 | - | 79.05% | 10-epoch baseline evaluated on the CIFAR-10 test split. |
| ResNet20 | 0.27M | 52 | 90.24% | - | Best validation checkpoint; includes 5-fold summary. |
| VGG16-BN | 15.2M | 86 | 92.46% | - | Best validation accuracy among available methods. |

ResNet20 5-fold mean test accuracy: 89.29% +/- 0.31 percentage points.

Generated files:
- best_accuracy_bar.png
- validation_accuracy_curves.png
- validation_loss_curves.png
- train_val_accuracy_gap.png
- resnet20_kfold_accuracy.png
- summary_table.csv
