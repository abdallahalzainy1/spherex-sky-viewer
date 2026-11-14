"""Run the tool on the six challenge targets against LIVE IRSA data and save a GIF for each.
Usage: python -m scripts.test_targets   (needs internet; add --demo to test offline)"""
import base64, io, sys
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

TARGETS = {"barnards_star": ("Barnard's Star", 3), "eps_indi": ("eps Ind", 12),
           "wise0855": ("08h55m10.831s -07d14m42.53s", 3), "rw_aur": ("RW Aur", 3), "r_dra": ("R Dra", 3)}
demo = "--demo" in sys.argv
out = Path("output"); out.mkdir(exist_ok=True)
client = TestClient(app, raise_server_exceptions=False)
for key, (target, size) in TARGETS.items():
    r = client.get("/api/frames", params={"target": target, "size": size, "demo": demo}, timeout=600)
    if r.status_code != 200:
        print(f"FAIL {key:14} {r.status_code} {r.json().get('detail')}"); continue
    j = r.json()
    ims = [Image.open(io.BytesIO(base64.b64decode(b))).resize((300, 300), Image.NEAREST) for b in j["frames"]]
    ims[0].save(out / f"{key}.gif", save_all=True, append_images=ims[1:], duration=250, loop=0)
    print(f"OK   {key:14} {j['n']} frames  RA {j['ra']:.4f} Dec {j['dec']:.4f}")
