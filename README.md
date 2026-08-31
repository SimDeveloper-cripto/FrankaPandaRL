# Overview for Door Manipulation Tasks

## Generalized Door Closing Task (v1) with Robosuite 1.5.1

```powershell
# on Windows11
# 1. dependency needed:python3.10-dev
python -m venv .venv
.\.venv\Scripts\activate 
pip install --no-deps -r requirements.txt 
pip install -r requirements.txt
# 2. copy mujoco.dll into venv\Lib\robosuite\utils
& .venv\Scripts\python.exe close_generalized/train_gen.py --play
```

```bash
# on Linux
# 1. dependency needed:python3.10-dev
python -m venv .venv
source ./.venv/bin/activate
pip install --no-deps -r requirements.txt 
pip install -r requirements.txt
# 2. copy mujoco.dll into venv\Lib\robosuite\utils
python close_generalized/train_gen.py --play
```

```bash
# on macOS
# 1. dependency needed:python3.10-dev
python3.10 -m venv .venv
source ./.venv/bin/activate
pip install --no-deps -r requirements_mac.txt
pip install -r requirements_mac.txt

# 2. see current best_model.zip and vecnormalize.pkl (results)
mjpython close_generalized/train_gen.py --play

# You can resume Training and add a new limit
./.venv/bin/python close_generalized/train_gen.py --resume --total-steps <new_steps_count>

# You can resume Training with Current best model and vecnormalize
./.venv/bin/python close_generalized/train_gen.py --resume-model runs/close_gen/best_model.zip --resume-vecnorm runs/close_gen/vecnormalize.pkl
```

---

## Generalization of the Generalized Door Closing Task (v2) with Robosuite 1.5.1

```bash
# Same .venv of close_generalized
# Of course it works fine also on Windows11 and Linux

# Start Training from scratch
mjpython close_generalized_v2/train_gen_v2.py

# Play with Current best model (viewer MuJoCo)
mjpython close_generalized_v2/train_gen_v2.py --play --model runs/close_gen_v2_curriculum_0_new_110626/best_model.zip
mjpython close_generalized_v2/train_gen_v2.py --play --model runs/close_gen_v2_curriculum_1_new_110626/best_model.zip

# Resume Training
./.venv/bin/python close_generalized_v2/train_gen_v2.py --resume

# Resume from specific checkpoint
./.venv/bin/python close_generalized_v2/train_gen_v2.py  \
    --resume-model runs/close_gen_v2/best_model.zip      \
    --resume-vecnorm runs/close_gen_v2/vecnormalize.pkl

# Add more steps to an already started Training
./.venv/bin/python close_generalized_v2/train_gen_v2.py --resume --total-steps <new_steps_count>
```

---

## Generalization of the Generalized Door Opening Task with Robosuite 1.5.1

```bash
# Same .venv of close_generalized
# Of course it works fine also on Windows11 and Linux

# Train from scratch (only curr 1)
# SR = 1 at 800k steps
mjpython open_generalized_v2/train_curriculum_v2 --total-steps 1500000

# Play (only curr 1)
mjpython open_generalized_v2/train_curriculum_v2 --play

# Run Test Suite
mjpython scratch/test_open_task_v2/run_all_tests.py --preset full
```

---

## Unified Reward Machine

```bash
# Same .venv of close_generalized
# Of course it works fine also on Windows11 and Linux

cd unified_door
export PROGETTI_ORIGINALI="$(cd .. && pwd)"

python3 train_unified.py --task close --total-steps 1000000
python3 train_unified.py --task open  --total-steps 1000000

python3 train_unified.py --task close --eval --episodes 20
python3 train_unified.py --task open  --eval --episodes 20

mjpython train_unified.py --task close --play
mjpython train_unified.py --task open  --play --slow 2
```


### Result Folders (trained models)

- [OK] Generalized Door Closing Task __runs/close_gen__
- [OK] Generalization of the Generalized Door Closing Task (curr 0) __runs/close_gen_v2_curriculum_0_new_110626/__
- [OK] Generalization of the Generalized Door Closing Task (curr 1) __runs/close_gen_v2_curriculum_1_new_110626/__
- [OK] Generalization of the Generalized Door Opening Task (curr 1) __runs/open_gen_v2/__
- [OK] Unified Reward Machine for both Tasks (curr 1) __unified_door/runs/__

### Test Results Folders (trained models)

- [OK] __scratch/__
- [OK] __unified_door/tests/__
- [OK] __figure/__