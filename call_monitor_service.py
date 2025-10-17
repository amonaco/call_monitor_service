from fastapi import FastAPI, Request, HTTPException
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, os
from datetime import datetime

# Config
AUTH_KEY = ""
MODEL_PATH = "autoencoder_model.pt"
META_PATH = "prefix_data.json"

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(2, input_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(2, input_dim // 2), 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, max(2, input_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(2, input_dim // 2), input_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# State management
app = FastAPI(title="call_monitor_service")

state = {
    "mode": "learning",      # or "policing"
    "prefix_dict": {},
    "threshold": None,
    "model": None,
    "optimizer": None,
    "criterion": nn.MSELoss(),
    "train_samples": []
}

def one_hot(prefix):
    """Dynamically one-hot encode prefix"""
    d = state["prefix_dict"]
    if prefix not in d:
        d[prefix] = len(d)
    idx = d[prefix]
    vec = np.zeros(len(d))
    vec[idx] = 1
    return vec

def rebuild_model():
    """Rebuild model if needed (e.g., prefix set grew)"""
    dim = len(state["prefix_dict"])
    state["model"] = Autoencoder(dim)
    state["optimizer"] = optim.Adam(state["model"].parameters(), lr=0.01)
    print(f"Model rebuilt with input_dim={dim}")

# Persistence (planning to use Redis next) 
def save_state():
    if not state["model"]:
        return
    torch.save(state["model"].state_dict(), MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "prefix_dict": state["prefix_dict"],
            "threshold": state["threshold"],
            "mode": state["mode"],
            "last_save": datetime.now().isoformat()
        }, f, indent=2)
    print("✅ State saved")

def load_state():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        print("No saved state found, starting fresh.")
        return
    with open(META_PATH) as f:
        meta = json.load(f)
        state["prefix_dict"] = meta["prefix_dict"]
        state["threshold"] = meta.get("threshold")
        state["mode"] = meta.get("mode", "learning")
    dim = len(state["prefix_dict"])
    state["model"] = Autoencoder(dim)
    state["model"].load_state_dict(torch.load(MODEL_PATH))
    state["model"].eval()
    state["optimizer"] = optim.Adam(state["model"].parameters(), lr=0.01)
    print("✅ State loaded")

load_state()

# FastAPI endpoints
@app.get("/status")
def get_status():
    return {
        "mode": state["mode"],
        "prefixes_learned": list(state["prefix_dict"].keys()),
        "threshold": state["threshold"],
        "prefix_count": len(state["prefix_dict"])
    }

@app.post("/admin/mode")
async def switch_mode(request: Request):
    data = await request.json()
    if data.get("auth_key") != AUTH_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    mode = data.get("mode")
    if mode not in ["learning", "policing"]:
        raise HTTPException(status_code=400, detail="Invalid mode")
    state["mode"] = mode
    if mode == "policing":
        # compute threshold if not already set
        if state["threshold"] is None and state["train_samples"]:
            with torch.no_grad():
                x = torch.tensor(np.array(state["train_samples"]), dtype=torch.float32)
                recon = state["model"](x)
                errs = torch.mean(torch.abs(recon - x), dim=1).numpy()
                state["threshold"] = float(np.mean(errs) + 2*np.std(errs))
        save_state()
    return {"message": f"Mode switched to {mode}"}

@app.post("/feed")
async def feed_prefix(request: Request):
    data = await request.json()
    prefix = str(data.get("prefix"))
    if not prefix:
        raise HTTPException(status_code=400, detail="Missing prefix")

    # Learning mode
    if state["mode"] == "learning":
        vec = one_hot(prefix)
        state["train_samples"].append(vec)
        if state["model"] is None or state["model"].encoder[0].in_features != len(state["prefix_dict"]):
            rebuild_model()
        x = torch.tensor(np.array(state["train_samples"]), dtype=torch.float32)
        state["optimizer"].zero_grad()
        out = state["model"](x)
        loss = state["criterion"](out, x)
        loss.backward()
        state["optimizer"].step()
        return {"status": "learning", "prefix": prefix, "loss": float(loss.item())}

    # Policing mode
    else:
        if prefix not in state["prefix_dict"]:
            return {"status": "alert", "prefix": prefix, "reason": "unknown_prefix"}
        vec = np.zeros(len(state["prefix_dict"]))
        vec[state["prefix_dict"][prefix]] = 1
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon = state["model"](x)
            error = torch.mean(torch.abs(recon - x)).item()
        if error > state["threshold"]:
            result = {"status": "alert", "prefix": prefix, "error": error}
        else:
            result = {"status": "ok", "prefix": prefix, "error": error}
        return result


# TODO: add auth here
@app.route('/forget', methods=['POST'])
def forget_prefixes():
    """
    Forget (remove) one or more prefixes from the learned state.
    Example payload:
    {
        "prefixes": ["hello", "how are"]
    }
    """
    data = request.get_json(force=True)
    prefixes = data.get("prefixes", [])

    if not isinstance(prefixes, list):
        return jsonify({"error": "prefixes must be a list"}), 400

    removed = []
    for prefix in prefixes:
        if prefix in state["prefix_dict"]:
            del state["prefix_dict"][prefix]
            removed.append(prefix)

    # Save immediately after forgetting
    save_state(state)

    return jsonify({
        "removed": removed,
        "remaining_prefixes": len(state["prefix_dict"])
    })
