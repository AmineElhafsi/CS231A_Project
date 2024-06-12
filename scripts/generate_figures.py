import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.scene.scene import Scene
from src.utils.io import load_yaml

# Poster Methods Figure
# load scene
# scene = Scene.from_directory("/home/amine/Documents/Research/gaussian-splatting-playground/data/runs/office0")
# scene.visualize_submap(0)

# Evaluation results

# get list of directories within path
saved_runs_directory = "/home/amine/Documents/Research/gaussian-splatting-playground/data/runs"
saved_runs = [os.path.join(saved_runs_directory, d) for d in os.listdir(saved_runs_directory)]

# aggregate saved results
all_results = {}
for run in tqdm(saved_runs):
    run_name = os.path.basename(run)
    run_results = {
        "frame_optimization_times": [],
        "frame_average_optimization_iteration_times": [],
        "frame_psnrs": [],
        "frame_ssims": [],
        "new_submap": [],
        "num_gaussians": [],
    }


    # extract results
    results_dir = os.path.join(run, "results")
    if not os.path.exists(results_dir):
        print("Results not found for run: ", run)
        continue
    
    results_files = glob.glob(os.path.join(results_dir, "results_*.ckpt"))
    results_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    for result_file in results_files:
        results = torch.load(result_file)
        breakpoint()
        run_results["frame_optimization_times"].append(results["total_optimization_time"])
        run_results["frame_average_optimization_iteration_times"].append(results["average_optimization_iteration_time"])
        run_results["frame_psnrs"].append(results["psnr_render"])
        run_results["frame_ssims"].append(results["ssim_render"])
        run_results["new_submap"].append(results["new_submap"])
        run_results["num_gaussians"].append(results["num_gaussians"])
    
    run_results["mean_frame_optimization_time"] = np.mean(run_results["frame_optimization_times"])
    run_results["stdev_frame_optimization_time"] = np.std(run_results["frame_optimization_times"])
    submap_init_times = []
    frame_time = []
    for i, init_submap in enumerate(run_results["new_submap"]):
        if init_submap:
            submap_init_times.append(run_results["frame_optimization_times"][i])
        else:
            frame_time.append(run_results["frame_optimization_times"][i])
    run_results["mean_submap_init_time"] = np.mean(submap_init_times)
    run_results["stdev_submap_init_time"] = np.std(submap_init_times)
    run_results["mean_frame_time"] = np.mean(frame_time)
    run_results["stdev_frame_time"] = np.std(frame_time)
    run_results["mean_iteration_time"] = np.mean(run_results["frame_average_optimization_iteration_times"])
    run_results["stdev_iteration_time"] = np.std(run_results["frame_average_optimization_iteration_times"])
    run_results["mean_psnr"] = np.mean(run_results["frame_psnrs"])
    run_results["stdev_psnr"] = np.std(run_results["frame_psnrs"])
    run_results["mean_ssim"] = np.mean(run_results["frame_ssims"])
    run_results["stdev_ssim"] = np.std(run_results["frame_ssims"])

    all_results[run_name] = run_results

for run_name in all_results.keys():
    print("Run: ", run_name)
    print("------------------------")
    print(" ")
    print("PSNR: ", all_results[run_name]["mean_psnr"], " +/- ", all_results[run_name]["stdev_psnr"])
    print("SSIM: ", all_results[run_name]["mean_ssim"], " +/- ", all_results[run_name]["stdev_ssim"])
    print("Submap Init Time: ", all_results[run_name]["mean_submap_init_time"], " +/- ", all_results[run_name]["stdev_submap_init_time"])
    print("Frame Time: ", all_results[run_name]["mean_frame_time"], " +/- ", all_results[run_name]["stdev_frame_time"])
    print(" ")
    print("------------------------")

# plot of runtime vs. frame
x1 = np.arange(len(all_results["office0"]["frame_optimization_times"]))
y1 = np.array(all_results["office0"]["frame_optimization_times"])

x2 = np.arange(len(all_results["rgbd_dataset_freiburg1_desk"]["frame_optimization_times"]))
y2 = np.array(all_results["rgbd_dataset_freiburg1_desk"]["frame_optimization_times"])

# Adjust the number of points in the plots
min_len = min(len(x1), len(x2))
x1 = x1[:min_len]
y1 = y1[:min_len]
x2 = x2[:min_len]
y2 = y2[:min_len]
new_submap_indices1 = [i for i, new_submap in enumerate(all_results["office0"]["new_submap"]) if (new_submap and i < min_len)]
new_submap_indices2 = [i for i, new_submap in enumerate(all_results["rgbd_dataset_freiburg1_desk"]["new_submap"]) if (new_submap and i < min_len)]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot for Office0
ax1.plot(x1, y1, linestyle='--', c='k', linewidth=0.5)
ax1.scatter(x1, y1, c='k', marker='o', s=10)
ax1.plot(x1[new_submap_indices1], y1[new_submap_indices1], 'r*')
ax1.set_xlabel("Frame Index")
ax1.set_ylabel("Frame Optimization Time (s)")
ax1.set_title("High Resolution Mapping (Replica Office0)")
ax1.grid(True, linestyle='--', axis='y')

# Plot for Freiburg1 Desk
ax2.plot(x2, y2, linestyle='--', c='k', linewidth=0.5)
ax2.scatter(x2, y2, c='k', marker='o', label="Mapping Iteration", s=10)
ax2.plot(x2[new_submap_indices2], y2[new_submap_indices2], 'r*', label="New Submap")
ax2.set_xlabel("Frame Index")
ax2.set_ylabel("Frame Optimization Time (s)")
ax2.set_title("Low Resolution Mapping (TUM RGBD Freiburg1 Desk)")
ax2.grid(True, linestyle='--', axis='y')

# Set uniform tick marks
max_x = max(len(x1), len(x2))
ax1.set_xticks(np.linspace(0, max_x, 6).astype(int))
ax2.set_xticks(np.linspace(0, max_x, 6).astype(int))

max_y = max(max(y1), max(y2))
ax1.set_yticks(np.linspace(0, max_y, 6).astype(int))
ax2.set_yticks(np.linspace(0, max_y, 6).astype(int))

# Move legends to overall figure
fig.legend(loc='lower center', ncol=2)

plt.suptitle("Mapping Computation Time")
plt.savefig("computation_time.png")

# start new plot (single, no subplots)
plt.figure()
# standard_indices1 = [i for i, new_submap in enumerate(all_results["office0"]["new_submap"]) if (not new_submap)]
# standard_indices2 = [i for i, new_submap in enumerate(all_results["office1"]["new_submap"]) if (not new_submap)]
# standard_indices3 = [i for i, new_submap in enumerate(all_results["office2"]["new_submap"]) if (not new_submap)]
# standard_indices4 = [i for i, new_submap in enumerate(all_results["rgbd_dataset_freiburg1_desk"]["new_submap"]) if (not new_submap)]
# standard_indices5 = [i for i, new_submap in enumerate(all_results["rgbd_dataset_freiburg2_xyz"]["new_submap"]) if (not new_submap)]
# standard_indices6 = [i for i, new_submap in enumerate(all_results["rgbd_dataset_freiburg3_long_office_household"]["new_submap"]) if (not new_submap)]
# plt.scatter(
#     np.array(all_results["office0"]["num_gaussians"])[standard_indices1], 
#     np.array(all_results["office0"]["frame_optimization_times"])[standard_indices1],
#     s=5,
#     c='r',
# )
# plt.scatter(
#     np.array(all_results["office1"]["num_gaussians"])[standard_indices2], 
#     np.array(all_results["office1"]["frame_optimization_times"])[standard_indices2],
#     s=5,
#     c='r',
# )
# plt.scatter(
#     np.array(all_results["office2"]["num_gaussians"])[standard_indices3], 
#     np.array(all_results["office2"]["frame_optimization_times"])[standard_indices3],
#     s=5,
#     c='r',
#     label="Replica"
# )
# plt.scatter(
#     np.array(all_results["rgbd_dataset_freiburg1_desk"]["num_gaussians"])[standard_indices4], 
#     np.array(all_results["rgbd_dataset_freiburg1_desk"]["frame_optimization_times"])[standard_indices4],
#     s=5,
#     c='b',
#     label="TUM RGBD"
# )
# plt.scatter(
#     np.array(all_results["rgbd_dataset_freiburg2_xyz"]["num_gaussians"])[standard_indices5], 
#     np.array(all_results["rgbd_dataset_freiburg2_xyz"]["frame_optimization_times"])[standard_indices5],
#     s=5,
#     c='b',
# )
# plt.scatter(
#     np.array(all_results["rgbd_dataset_freiburg3_long_office_household"]["num_gaussians"])[standard_indices6], 
#     np.array(all_results["rgbd_dataset_freiburg3_long_office_household"]["frame_optimization_times"])[standard_indices6],
#     s=5,
#     c='b',
#     label="TUM RGBD"
# )

plt.scatter(
    np.array(all_results["office0"]["num_gaussians"]), 
    np.array(all_results["office0"]["frame_average_optimization_iteration_times"])/1000,
    s=5,
    c='r',
)
plt.scatter(
    np.array(all_results["office1"]["num_gaussians"]), 
    np.array(all_results["office1"]["frame_average_optimization_iteration_times"])/1000,
    s=5,
    c='r',
)
plt.scatter(
    np.array(all_results["office2"]["num_gaussians"]), 
    np.array(all_results["office2"]["frame_average_optimization_iteration_times"])/1000,
    s=5,
    c='r',
    label="Replica"
)
plt.scatter(
    np.array(all_results["rgbd_dataset_freiburg1_desk"]["num_gaussians"]), 
    np.array(all_results["rgbd_dataset_freiburg1_desk"]["frame_average_optimization_iteration_times"])/1000,
    s=5,
    c='b',
    label="TUM RGBD"
)
plt.xlabel("Number of Gaussians")
plt.ylabel("Avg. Optimization Iteration Time (ms)")
plt.grid(True, linestyle='--')
plt.legend()
plt.title("Computation Time Scaling with Map Size")
plt.savefig("computation_time_scaling.png")

torch.save(all_results, "/home/amine/Documents/Research/gaussian-splatting-playground/data/all_results.ckpt")




