import json
from matplotlib import pyplot as plt

model_params = {
    "EleutherAI/pythia-410m-v0": 302_311_424,
    "EleutherAI/pythia-1b-v0": 805_736_448,
    "EleutherAI/pythia-1.4b-v0": 1_208_602_624,
    "EleutherAI/pythia-2.8b-v0": 2_517_652_480,
}

with open("transfer_results.json") as f:
    results = json.load(f)

xs = []
ys = []
zs = []

for model, result in results.items():
    xs.append(model_params[model])
    ys.append(result["strong_transfer_acc"])
    zs.append(result["strong_gt_acc"])

# sort xs and ys and zs by xs
xs, ys, zs = zip(*sorted(zip(xs, ys, zs)))

plt.plot(xs, ys, "o-")
plt.plot(xs, zs, "o-")
plt.xscale("log")
plt.xlabel("Number of non-embed params")
plt.ylabel("Strong transfer accuracy")

plt.show()