# mini-WAM

A tiny World Action Model from scratch for robot learning.

mini-WAM extends the educational philosophy of mini-VLA.  

Where mini-VLA learns:

```text
image_t + language + state_t → action_t
```

mini-WAM learns:

```text
image_t + language + state_t → action_t + state_t+1 + image_t+1
```

## Why mini-WAM?

A VLA learns to act.
A WAM learns to act while predicting what its action will do to the world.

## Quickstart

```sh
pip install -r requirements.txt

python scripts/collect_data.py --env toy --num_episodes 100

python scripts/train.py \
  --env toy \
  --model_type wam \
  --action_head mlp \
  --predict_state \
  --device cpu
  ```