"""Single-card identity and hotspot checks inside the benchmark container."""
import json
import os
import time
from pathlib import Path

import torch

BOARD = Path("/board")
HWMON, = (BOARD / "hwmon").glob("hwmon*")


def temperature():
    return int((HWMON / "temp2_input").read_text()) / 1000


def identify():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    p = torch.cuda.get_device_properties(0)
    bus = int(os.environ["PCI"].split(":")[1], 16)
    if torch.cuda.device_count() != 1 or p.pci_bus_id != bus:
        raise RuntimeError(f"Expected one GPU on PCI bus {bus}, found {p}")
    if p.gcnArchName != "gfx1201" or "R9600D" not in p.name:
        raise RuntimeError(f"This reproduction requires a gfx1201 R9600D, found {p}")
    return dict(uuid=str(p.uuid), pci=os.environ["PCI"], name=p.name, arch=p.gcnArchName,
                torch=torch.__version__, hip=torch.version.hip,
                power_cap_W=int((HWMON / "power1_cap").read_text()) / 1e6)


if __name__ == "__main__":
    card = identify()
    time.sleep(0.4)
    print(json.dumps(dict(**card, baseline_C=temperature())))
