
To train a model:

```markdown
python main.py --train 
               --dataset # choice from cifar10, mnist 
               --[add'l options]
```

To evaluate a model or generate from a model:

```markdown
python main.py --generate # [--evaluate]; if evaluate, need to specify dataset and data_path 
               --restore_file # path to .pt checkpoint
```

To generate the resultant graphs in the paper:

```markdown
python LLR.py
```

