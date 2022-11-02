# Training Vision-Language Models with Less Bimodal Supervision

This repository contains the official code of the paper: ["Training Vision-Language Models with Less Bimodal Supervision
"](https://arxiv.org/abs/2211.00262).

## NOTE: The training commands to reproduce the experiments from the paper will be added soon, along with code to generate the tables and the figures.  
Also, documentation for general usage of the code will be added.

***

## Setup
### Requirements
```
conda create -n lbs python=3.9
conda activate lbs
pip install -r dev-requirements.txt requirements.txt
```

### Data
See [here](https://github.com/eladsegal/less-bimodal-sup/tree/master/data).

***

## Training
The general form for running a training command is:
```
python src/train.py main_config.jsonnet --u additional_config.jsonnet
```

### Examples
#### Finetuning on VQAv2
```
python src/train.py configs/less-bimodal-sup/vl_finetuning.jsonnet --u configs/data/vqa.jsonnet
```

#### Pretraining
```
python src/train.py configs/less-bimodal-sup/vl_pretraining.jsonnet --u configs/data/conceptual_captions.jsonnet datamodule.batch_size=48 trainer.accumulate_grad_batches=10 trainer.gpus=8
```

***

#### Training Continuation
##### Resume from the last checkpoint
```
python src/train.py ../outputs/experiment/checkpoints/last.ckpt
```
OR
```
python src/train.py ../outputs/experiment
```

#### Start a new training and initialize weights from a checkpoint
```
python src/train.py main_config.jsonnet --u additional_config.jsonnet load_weights=../outputs/exp/checkpoints/some_checkpoint.ckpt
```

***

### View Compiled Jsonnet
#### Jsonnet Config
Use args like in [training](#training)
```
python tools/jsonnet.py [args]
```
#### Regular Jsonnet
```
python tools/jsonnet.py some_file.jsonnet --simple
```

***

### Hydra Command-line Tips
- You can modify a list item by using its index in a command line argument (e.g. trainer.logger.0.notes=something)
- You can modify a top-level jsonnet field before the json stage by using a command line argument prefixed with "o__" 

***

## Citation
```
@inproceedings{
    segal2022training,
    title={Training Vision-Language Models with Less Bimodal Supervision},
    author={Elad Segal and Ben Bogin and Jonathan Berant},
    booktitle={4th Conference on Automated Knowledge Base Construction},
    year={2022}
}
```
