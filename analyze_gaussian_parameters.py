import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state
from utils.render_utils import get_state_at_time
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def analyze_and_plot(dataset: ModelParams, hyperparam, iteration: int, pipeline: PipelineParams,
                     target_frames: list, rois: list, opacity_threshold: float):
    """
    Analyzes Gaussian parameters for a specific set of frames and ROIs, then plots a bar chart.

    Args:
        dataset (ModelParams): Dataset parameters.
        hyperparam: Hyperparameters for the Gaussian model.
        iteration (int): The iteration number of the model to load.
        pipeline (PipelineParams): Pipeline parameters.
        target_frames (list): A list of frame indices to analyze.
        rois (list): A list of ROIs, one for each target frame.
        opacity_threshold (float): The opacity threshold to filter Gaussians.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Use test cameras for analysis, can be changed to getVideoCameras()
        viewpoints = scene.getTestCameras()
        if not viewpoints:
            print("No cameras found to analyze.")
            return

        # NEW LOGIC: Analyze specified frames with their corresponding ROIs.
        print(f"Analyzing frames {target_frames}...")
        avg_volumes = []
        avg_anisotropies = []

        for i, frame_idx in enumerate(tqdm(target_frames, desc="Analyzing frames")):
            if frame_idx >= len(viewpoints):
                print(f"Warning: frame index {frame_idx} is out of bounds. Skipping.")
                continue

            viewpoint = viewpoints[frame_idx]
            roi = rois[i]
            
            # Get Gaussian state at the current viewpoint's time
            means3D, scales_t, _, opacities, _ = get_state_at_time(gaussians, viewpoint)

            # --- Identify Gaussians in ROI for the CURRENT frame ---
            w_T_c = viewpoint.world_view_transform
            c_T_w = torch.inverse(w_T_c).to(means3D.device)
            ones = torch.ones(means3D.shape[0], 1, device=means3D.device)
            points_homogeneous = torch.cat([means3D, ones], dim=1)
            points_camera_space = (c_T_w @ points_homogeneous.T).T[:, :3]

            opacity_mask = opacities.squeeze(-1) > opacity_threshold

            x_min, y_min, x_max, y_max = roi
            img_w, img_h = viewpoint.image_width, viewpoint.image_height
            ndc_x_min, ndc_y_min = 2.0 * x_min / img_w - 1.0, 2.0 * y_min / img_h - 1.0
            ndc_x_max, ndc_y_max = 2.0 * x_max / img_w - 1.0, 2.0 * y_max / img_h - 1.0
            
            z_camera = points_camera_space[:, 2]
            viewspace_points_x = points_camera_space[:, 0] / (z_camera + 1e-9)
            viewspace_points_y = points_camera_space[:, 1] / (z_camera + 1e-9)
            
            roi_mask = (viewspace_points_x >= ndc_x_min) & (viewspace_points_x <= ndc_x_max) & \
                       (viewspace_points_y >= ndc_y_min) & (viewspace_points_y <= ndc_y_max) & \
                       (z_camera > 0)
            
            target_mask = opacity_mask & roi_mask
            target_indices = torch.where(target_mask)[0]
            # --- End: Identification for the CURRENT frame ---

            if len(target_indices) == 0:
                print(f"Warning: No Gaussians found in ROI {roi} for frame {frame_idx}. Appending 0.")
                avg_volumes.append(0)
                avg_anisotropies.append(0)
                continue
            
            # Filter for target gaussians
            target_scales = scales_t[target_indices]

            # Calculate Volume and Anisotropy
            volumes = torch.prod(target_scales, dim=1)
            avg_volume = torch.mean(volumes).item()
            avg_volumes.append(avg_volume)

            anisotropy = torch.std(target_scales, dim=1)
            avg_anisotropy = torch.mean(anisotropy).item()
            avg_anisotropies.append(avg_anisotropy)

        # Plotting the results as a bar chart
        print("Plotting results as a bar chart...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        x_labels = [str(f) for f in target_frames]
        x_pos = np.arange(len(x_labels))
        
        # Plot Average Volume
        ax1.bar(x_pos, avg_volumes, align='center', alpha=0.7, color='b')
        ax1.set_ylabel('Average Volume')
        ax1.set_title(f'Gaussian Parameter Comparison for Frames {target_frames}')
        ax1.grid(True)

        # Plot Average Anisotropy
        ax2.bar(x_pos, avg_anisotropies, align='center', alpha=0.7, color='g')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Average Anisotropy')
        ax2.grid(True)

        plt.tight_layout()
        output_path = os.path.join(dataset.model_path, "analysis")
        os.makedirs(output_path, exist_ok=True)
        
        # Flatten the list of ROIs for the filename
        flat_rois = [coord for roi_coords in rois for coord in roi_coords]
        plot_filename = os.path.join(output_path, f"params_comparison_frames_{'_'.join(map(str, target_frames))}_rois_{'_'.join(map(str, flat_rois))}.png")
        plt.savefig(plot_filename)
        print(f"Analysis plot saved to: {plot_filename}")
        plt.close()

if __name__ == "__main__":
    parser = ArgumentParser(description="Analysis script for 4D Gaussian Splatting.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration number to load model.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, required=True, help="Path to the configuration file.")
    
    # Arguments for analysis
    parser.add_argument("--target_frames", type=int, nargs='+', required=True, help="A list of frame indices to analyze (e.g., 334 335 336).")
    parser.add_argument("--rois", type=int, nargs='+', required=True, help="A flattened list of ROIs, 4 values per frame (e.g., x1 y1 x2 y2 x1 y1 x2 y2 ...).")
    parser.add_argument("--opacity_threshold", type=float, default=0.1, help="Opacity threshold to filter Gaussians.")

    args = get_combined_args(parser)
    
    print("Running analysis for:", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(args.quiet)

    if len(args.rois) != len(args.target_frames) * 4:
        raise ValueError("The number of values for --rois must be 4 times the number of --target_frames.")

    # Reshape the flattened ROIs list into a list of lists
    rois_list = [args.rois[i:i + 4] for i in range(0, len(args.rois), 4)]

    analyze_and_plot(model.extract(args),
                     hyperparam.extract(args),
                     args.iteration,
                     pipeline.extract(args),
                     args.target_frames,
                     rois_list,
                     args.opacity_threshold)

    print("\nAnalysis complete.")