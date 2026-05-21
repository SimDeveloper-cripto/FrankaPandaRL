# Overview

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

# Resume Training and add new limit
./.venv/bin/python close_generalized/train_gen.py --resume --total-steps <new_steps_to_add>

# Resume Training with current best model and vecnormalize
./.venv/bin/python close_generalized/train_gen.py --resume-model runs/close_gen/best_model.zip --resume-vecnorm runs/close_gen/vecnormalize.pkl
```

## Results

Coming soon !!

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.