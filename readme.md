```markdown
To train a model:

```
python main.py --train --dataset # choice from cifar10, mnist --[add'l options]
```

To evaluate a model or generate from a model:

```
python main.py --generate # [--evaluate]; if evaluate, need to specify dataset and data_path --restore_file # path to .pt checkpoint
```

To generate the resultant graphs in the paper:

```
python LLR.py
```

cifar10 vs cifar100: llr mean: -2833.5894 ood llr mean: -633.9239 Correlation coefficient: -0.003920205743089568, P-value: 0.695078108253184 auc: 1 - 0.479697285 auc_llr: 1 - 0.00012937999999995675

cifar10 vs svhn: llr mean: -2833.8066 ood llr mean: -447.9810 Correlation coefficient: 0.0006811285793404708, P-value: 0.9457025666158174 auc: 1 - 0.887750845 auc_llr: 1 - 0.03661370000000019

mnist vs fasionmnist: llr mean: -428.9050 ood llr mean: -65.3371 Correlation coefficient: 0.002028172762088479, P-value: 0.839297508592875 auc: 1 - 0.0002914799999998996 auc_llr: 1 - 0.00015555999999994352
```
