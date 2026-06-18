import pickle
from stable_baselines3 import SAC

model_path = ""
vn_path    = ""

print("Loading model...")
try:
    model = SAC.load(model_path)
    print("Model loaded successfully!")
    print("Observation space:", model.observation_space)
    print("Action space:"     , model.action_space)
    print("Policy Class:"     , type(model.policy))
    print("Net arch:"         , model.policy_kwargs)
except Exception as e:
    print("Error loading model:", e)

print("\nLoading VecNormalize...")
try:
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    print("VecNormalize loaded successfully!")
    print("Type:", type(vn))
    if hasattr(vn, "obs_rms"):
        print("obs_rms mean shape:", vn.obs_rms.mean.shape)
        print("obs_rms var shape:" , vn.obs_rms.var.shape)
    else:
        print("No obs_rms attribute found. Directory of object:")
        print(dir(vn))
except Exception as e:
    print("Error loading VecNormalize:", e)