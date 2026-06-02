## Generalized Door Closing Task with Robosuite 1.5.1

```powershell
# Windows11
# 1. dependency needed:python3.10-dev
python -m venv .venv
.\.venv\Scripts\activate 
pip install --no-deps -r requirements.txt 
pip install -r requirements.txt
# 2. copy mujoco.dll into venv\Lib\robosuite\utils 
& .venv\Scripts\python.exe close_generalized/train_gen.py --play
```

```bash
# Linux
# 1. dependency needed:python3.10-dev
python -m venv .venv
source ./.venv/bin/activate
pip install --no-deps -r requirements.txt 
pip install -r requirements.txt
# 2. copy mujoco.dll into venv\Lib\robosuite\utils 
python close_generalized/train_gen.py --play
```

```bash
# macOS
# 1. dependency needed:python3.10-dev
python3.10 -m venv .venv
source ./.venv/bin/activate
pip install --no-deps -r requirements_mac.txt
pip install -r requirements_mac.txt

# 2. see current best_model.zip and vecnormalize.pkl
mjpython close_generalized/train_gen.py --play

# Resume Training and Add new limit
./.venv/bin/python close_generalized/train_gen.py --resume --total-steps <new_steps_to_add>

# Resume Training with Current best model and vecnormalize
./.venv/bin/python close_generalized/train_gen.py --resume-model runs/close_gen/best_model.zip --resume-vecnorm runs/close_gen/vecnormalize.pkl
```

### Results

Coming soon !!

## Generalization of the Generalized Door Closing Task with Robosuite 1.5.1

```bash
# macOS — same .venv of close_generalized

# Training from scratch
./.venv/bin/python close_generalized_v2/train_gen_v2.py

# Play with Current best model (viewer MuJoCo)
mjpython close_generalized_v2/train_gen_v2.py --play --model runs/close_gen_v2/best_model.zip

# Resume training
./.venv/bin/python close_generalized_v2/train_gen_v2.py --resume

# Resume from specific checkpoint
./.venv/bin/python close_generalized_v2/train_gen_v2.py  \
    --resume-model runs/close_gen_v2/best_model.zip      \
    --resume-vecnorm runs/close_gen_v2/vecnormalize.pkl

# Add more steps to an already started training
./.venv/bin/python close_generalized_v2/train_gen_v2.py --resume --total-steps <new_steps_to_add>

# beta-networks
./.venv/bin/python close_generalized_v2/train_gen_v2.py --beta-net
```

### Results

Coming soon !!

## Generalized Door Opening Task with Robosuite 1.5.1

```bash
# macOS — same .venv of close_generalized

# Training from scratch
./.venv/bin/python open_generalized/train_curriculum.py

# Play with Current best model (viewer MuJoCo)
mjpython open_generalized/train_curriculum.py --play

# Play with specific model
mjpython open_generalized/train_curriculum.py --play --model runs/open_gen/best_model.zip

# Resume training
./.venv/bin/python open_generalized/train_curriculum.py --resume

# Resume from specific checkpoint
./.venv/bin/python open_generalized/train_curriculum.py  \
    --resume-model runs/open_gen/best_model.zip          \
    --resume-vecnorm runs/open_gen/vecnormalize.pkl

# Add more steps to an already started training
./.venv/bin/python open_generalized/train_curriculum.py --resume --total-steps <new_steps_to_add>
```

### TODO

- [ ] Generalize this Generalized Task

### Results

Coming soon !!


## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.