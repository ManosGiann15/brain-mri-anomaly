import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.utilities.file_and_folder_operations import load_json

# --- paths you already have after nnUNetv2 planning/preprocessing ---
plans_path = "/path/to/nnUNetPlans.json"
dataset_json_path = "/path/to/dataset.json"
configuration = "3d_fullres"          # or "2d", "3d_lowres", etc. (use what your data supports)

# --- load plans & dataset metadata ---
plans = load_json(plans_path)
dataset_json = load_json(dataset_json_path)
pm = PlansManager(plans)
cm = ConfigurationManager(pm, configuration)

# --- channels: make output = input for autoencoding ---
in_channels = 1          # e.g., 1 for FLAIR; 4 if you stack T1/T1ce/T2/FLAIR
out_channels = in_channels

# --- build the nnU-Net U-Net with no deep supervision ---
net = get_network_from_plans(
    plans_manager=pm,
    configuration_manager=cm,
    dataset_json=dataset_json,
    num_input_channels=in_channels,
    num_output_channels=out_channels,
    deep_supervision=False,     # important for reconstruction
)

# now 'net' is a plain PyTorch nn.Module (U-Net). Train it with a reconstruction loss.
x = torch.randn(1, in_channels, *cm.patch_size)   # test forward
y_hat = net(x)                                     # y_hat has same channels as x
