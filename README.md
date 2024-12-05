# squishy-bagels

## How to run

### Install required python packages

``` bash
pip install -r mcvd-pytorch/requirements.txt
```

### Run model over videos

``` bash
python main.py --config configs/snake.yaml --data_path datasets/snake/videos --exp ../output
```

## Files of interest within mcvd-pytorch

- ./mcvd-pytorch/configs/snake.yaml (Our config for the snake dataset)
- ./mcvd-pytorch/datasets/snake.py (Processes snake dataset to be compatible with pytorch)
- ./mcvd-pytorch/datasets/snake/videos (The source of our data - 16 frame videos of snake gameplay)
- ./mcvd-pytorch/main.py (Where the app runs)
- ./mcvd-pytorch/runners/ncsn_runner.py (Does the heavy lifting - model training)
