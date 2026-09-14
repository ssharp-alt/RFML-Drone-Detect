import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
import gc
import warnings
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Limit memory usage for matplotlib
plt.rcParams['figure.max_open_warning'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


class DroneDataset(Dataset):
    """
    Dataset class for drone IQ Signals + transform to spectrogram
    """
    def __init__(self, path, transform=None, device=None):
        self.path = path
        self.files = os.listdir(path)
        self.files = [f for f in self.files if f.endswith('pt')] # filter for files with .pt extension  
        self.files = [f for f in self.files if f.startswith('IQdata_sample')] # filter for files which start with IQdata_sample in name
        self.transform = transform
        self.device = device

        # create list of targets and snrs for all samples
        self.targets = []
        self.snrs = []
        
        for file in self.files:
            self.targets.append(int(file.split('_')[2][6:])) # get target from file name
            self.snrs.append(int(file.split('_')[3].split('.')[0][3:])) # get snr from file name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file = self.files[idx]
            sample_id = int(file.split('_')[1][6:]) # get sample id from file name
            
            # More robust file loading
            try:
                data_path = os.path.join(self.path, file)
                # Try with weights_only=True first (safer)
                try:
                    data_dict = torch.load(data_path, weights_only=True, map_location='cpu')
                except:
                    # Fall back to default loading if needed
                    data_dict = torch.load(data_path, map_location='cpu')
                    
                # Verify data structure
                if not isinstance(data_dict, dict):
                    print(f"Warning: File {file} didn't load as expected dictionary, got {type(data_dict)}")
                    # Create a placeholder dictionary
                    data_dict = {'x_iq': torch.zeros(2, 1024), 'y': torch.tensor(0), 'snr': torch.tensor(0)}
                
                # Check required keys
                required_keys = ['x_iq', 'y', 'snr']
                for key in required_keys:
                    if key not in data_dict:
                        print(f"Warning: Missing key {key} in file {file}")
                        if key == 'x_iq':
                            data_dict[key] = torch.zeros(2, 1024)
                        elif key in ['y', 'snr']:
                            data_dict[key] = torch.tensor(0)
                            
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                # Create dummy data
                data_dict = {'x_iq': torch.zeros(2, 1024), 'y': torch.tensor(0), 'snr': torch.tensor(0)}
            
            iq_data = data_dict['x_iq']
            act_target = data_dict['y']
            act_snr = data_dict['snr']
            
            # Ensure data has correct dimensions
            if not isinstance(iq_data, torch.Tensor):
                print(f"Warning: iq_data in {file} is not a tensor, converting...")
                iq_data = torch.tensor(iq_data, dtype=torch.float32)
                
            # Make sure data has the right shape
            if len(iq_data.shape) < 2 or iq_data.shape[0] != 2:
                print(f"Warning: Incorrect shape for iq_data in {file}: {iq_data.shape}, reshaping...")
                # Try to reshape or recreate with zeros
                try:
                    if len(iq_data.shape) == 1:
                        # Assume it's a flat array that needs reshaping
                        length = iq_data.shape[0] // 2
                        iq_data = iq_data.reshape(2, length)
                    else:
                        # Create zeros
                        iq_data = torch.zeros(2, 1024, dtype=torch.float32)
                except:
                    iq_data = torch.zeros(2, 1024, dtype=torch.float32)

            # Process with transform if provided
            if self.transform:
                if self.device:
                    try:
                        iq_data = iq_data.to(device=self.device)
                    except Exception as e:
                        print(f"Error moving data to device: {e}")
                        iq_data = torch.zeros(2, 1024, dtype=torch.float32)
                        
                try:
                    transformed_data = self.transform(iq_data)
                except Exception as e:
                    print(f"Transform error: {e}")
                    transformed_data = None
            else:
                transformed_data = None

            return iq_data, act_target, act_snr, sample_id, transformed_data
            
        except Exception as e:
            print(f"Critical error in __getitem__ for index {idx}: {e}")
            # Return dummy data as fallback
            dummy_iq = torch.zeros(2, 1024, dtype=torch.float32)
            dummy_target = torch.tensor(0)
            dummy_snr = torch.tensor(0)
            dummy_id = -1
            return dummy_iq, dummy_target, dummy_snr, dummy_id, None
    
    def get_targets(self): # return list of targets
        return self.targets

    def get_snrs(self): # return list of snrs
        return self.snrs
    
    def get_files(self):
        return self.files
        
        
class SpectrogramTransform(torch.nn.Module):
    def __init__(
        self,
        device,
        n_fft=1024,
        win_length=1024,
        hop_length=1024,
        window_fn=torch.hann_window,
        power=None,
        normalized=False,
        center=False,
        onesided=False
    ):
        super().__init__()
        self.spec = Spectrogram(
            n_fft=n_fft, 
            win_length=win_length, 
            hop_length=hop_length, 
            window_fn=window_fn, 
            power=power, 
            normalized=normalized, 
            center=center, 
            onesided=onesided
        ).to(device=device)   
        self.win_length = win_length
        self.n_fft = n_fft
        self.device = device

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        try:
            # Check input dimensions
            if iq_signal is None or not isinstance(iq_signal, torch.Tensor):
                print(f"Warning: iq_signal is not a tensor or is None")
                return torch.zeros((2, self.n_fft, self.n_fft), device=self.spec.device)
                
            if len(iq_signal.shape) != 2 or iq_signal.shape[0] != 2:
                print(f"Warning: Unexpected shape {iq_signal.shape}, should be (2, N)")
                # Try to fix or return zeros
                if len(iq_signal.shape) == 2 and iq_signal.shape[1] == 2:
                    # Might be transposed
                    iq_signal = iq_signal.t()
                elif len(iq_signal.shape) == 1:
                    # Single dimension, try to reshape
                    half_len = iq_signal.shape[0] // 2
                    iq_signal = iq_signal.reshape(2, half_len)
                else:
                    # Can't fix, return zeros
                    return torch.zeros((2, self.n_fft, self.n_fft), device=self.spec.device)
            
            # Convert to complex signal
            try:
                complex_signal = iq_signal[0,:] + (1j * iq_signal[1,:])
                
                # Check for NaN or inf values
                if torch.isnan(iq_signal).any() or torch.isinf(iq_signal).any():
                    print("Warning: NaN or Inf values in input signal, replacing with zeros")
                    complex_signal = torch.zeros_like(complex_signal)
            except Exception as e:
                print(f"Error creating complex signal: {e}")
                # Return zeros on error
                return torch.zeros((2, self.n_fft, self.n_fft), device=self.spec.device)
                
            # Compute spectrogram
            try:
                spec = self.spec(complex_signal)
            except Exception as e:
                print(f"Spectrogram computation error: {e}")
                # Return zeros on error
                return torch.zeros((2, self.n_fft, self.n_fft), device=self.spec.device)
                
            # Convert to real representation
            try:
                spec = torch.view_as_real(spec)  # Returns a view of a complex input as a real tensor
                spec = torch.moveaxis(spec, 2, 0)  # move channel dimension to first dimension (1024, 1024, 2) -> (2, 1024, 1024)
            except Exception as e:
                print(f"Error in tensor conversion: {e}")
                # Return zeros on error
                return torch.zeros((2, self.n_fft, self.n_fft), device=self.spec.device)
                
            # Normalize
            spec = spec / self.win_length  # normalize by fft window size
            
            return spec
            
        except Exception as e:
            print(f"Unhandled error in SpectrogramTransform.forward: {e}")
            # Return zeros as fallback
            return torch.zeros((2, self.n_fft, self.n_fft), device=self.spec.device)


def generate_clean_spectrogram(spectrogram_2d, n_fft=1024, save_path=None, 
                              colormap='viridis', normalize=True):
    """
    Robust version to generate clean power spectrum image without axes, titles, or labels
    Fixed to avoid matplotlib errors and handle edge cases.
    """
    try:
        # Validate input
        if spectrogram_2d is None:
            print("Warning: Input spectrogram is None")
            return None
            
        # Validate spectrogram shape
        if not isinstance(spectrogram_2d, np.ndarray):
            print(f"Warning: spectrogram_2d is not a numpy array, but {type(spectrogram_2d)}")
            try:
                spectrogram_2d = np.array(spectrogram_2d)
            except:
                print("Could not convert to numpy array")
                return None
                
        if len(spectrogram_2d.shape) < 3:
            print(f"Warning: unexpected spectrogram shape: {spectrogram_2d.shape}")
            return None
            
        # Make sure dimensions are appropriate
        if spectrogram_2d.shape[0] != 2:
            print(f"Warning: First dimension should be 2, got {spectrogram_2d.shape}")
            # Try to reshape or transpose if possible
            if len(spectrogram_2d.shape) == 3 and spectrogram_2d.shape[2] == 2:
                # Maybe channels last format
                spectrogram_2d = np.transpose(spectrogram_2d, (2, 0, 1))
            else:
                return None
                
        # Validate array sizes to prevent index errors
        if spectrogram_2d.shape[1] < 2 or spectrogram_2d.shape[2] < 2:
            print(f"Error: Spectrogram dimensions too small: {spectrogram_2d.shape}")
            return None
            
        # FFT-shift to center frequencies - use try/except to avoid index errors
        try:
            spectrogram_2d = np.roll(spectrogram_2d, n_fft//2, axis=1)
        except Exception as e:
            print(f"FFT-shift error: {e}, skipping shift")
        
        # Calculate power spectrum (log10 of magnitude)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        try:
            # Use numpy broadcasting with bounds checking to prevent index errors
            real_part = spectrogram_2d[0,:,:].copy()
            imag_part = spectrogram_2d[1,:,:].copy()
            
            # Validate array values
            real_part = np.nan_to_num(real_part, nan=0.0, posinf=0.0, neginf=0.0)
            imag_part = np.nan_to_num(imag_part, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate magnitude
            magnitude = np.sqrt(real_part**2 + imag_part**2) + epsilon
            power_spec = np.log10(magnitude)
            
            # Extra check for NaN or inf values
            if np.isnan(power_spec).any() or np.isinf(power_spec).any():
                print("Warning: NaN or Inf values in power spectrum, replacing with zeros")
                power_spec = np.nan_to_num(power_spec, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"Error calculating power spectrum: {e}")
            return None
        
        # Normalize if requested
        if normalize:
            try:
                min_val = np.nanmin(power_spec)
                max_val = np.nanmax(power_spec)
                if max_val > min_val:  # Avoid division by zero
                    power_spec = (power_spec - min_val) / (max_val - min_val)
                    # Replace any remaining NaNs with zeros
                    power_spec = np.nan_to_num(power_spec)
            except Exception as e:
                print(f"Error normalizing: {e}")
        
        # Use a simpler approach to avoid 'apply_aspect' errors
        plt.ioff()  # Turn off interactive mode
        plt.close('all')  # Close all previous figures
        
        # Create new figure with fixed ratio
        fig = plt.figure(figsize=(5, 5), dpi=100)
        
        # Add a normal subplot (don't use custom Axes)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_position([0, 0, 1, 1])  # Make it fill the entire figure
        ax.set_axis_off()
        
        # Plot the image
        try:
            ax.imshow(power_spec, cmap=colormap)
            # Make sure figure is drawn
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error plotting image: {e}")
            plt.close(fig)
            return None
        
        if save_path:
            try:
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                    
                # Save with simpler parameters
                fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
            except Exception as e:
                print(f"Error saving figure: {e}")
                return None
            finally:
                plt.close(fig)  # Close the figure
                plt.close('all')  # Make sure all figures are closed
                gc.collect()  # Force garbage collection
            
            return save_path
        else:
            plt.close(fig)
            return None
    
    except Exception as e:
        print(f"Error in generate_clean_spectrogram: {e}")
        plt.close('all')  # Make sure to close all figures on error
        gc.collect()
        return None


def process_sample_batch(batch_indices, dataset, output_path, class_names, colormap, normalize, n_fft):
    """Process a batch of samples to reduce memory overhead"""
    results = []
    
    for idx in batch_indices:
        try:
            # Get sample data
            iq_data, target, snr, sample_id, transformed_data = dataset[idx]
            target_idx = target.item()
            snr_val = snr.item()
            
            # Get class name
            class_name = class_names[target_idx] if target_idx < len(class_names) else f"unknown_class_{target_idx}"
            
            # Define save directory
            save_dir = output_path / class_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Define image filename
            img_filename = f"sample_{sample_id}_snr_{snr_val}.png"
            save_path = save_dir / img_filename
            
            # Skip if file already exists
            if save_path.exists():
                results.append((target_idx, snr_val))
                continue
                
            # Generate and save clean power spectrum image
            if transformed_data is not None:
                # Handle potential errors in transform shape
                try:
                    # Move to CPU and convert to numpy safely
                    np_data = transformed_data.cpu().numpy()
                    
                    # Check if data has expected shape
                    if np_data.shape[0] != 2 or len(np_data.shape) != 3:
                        print(f"Warning: Unexpected shape {np_data.shape} for sample {idx}, reshaping...")
                        # Try to reshape if possible
                        if len(np_data.shape) >= 2:
                            if np_data.shape[0] == 1 and np_data.shape[1] == 2:
                                # Reshape to expected format
                                np_data = np_data.reshape(2, np_data.shape[2], np_data.shape[3] if len(np_data.shape) > 3 else 1)
                    
                    # Extra validation for dimensions
                    if len(np_data.shape) == 3 and np_data.shape[0] == 2 and np_data.shape[1] > 1 and np_data.shape[2] > 1:
                        # Generate spectrogram
                        success = generate_clean_spectrogram(
                            np_data,
                            n_fft=n_fft,
                            save_path=save_path,
                            colormap=colormap,
                            normalize=normalize
                        )
                        
                        # If successful, add to results
                        if success:
                            results.append((target_idx, snr_val))
                    else:
                        print(f"Error: Invalid shape for spectrogram data: {np_data.shape}")
                    
                except Exception as e:
                    print(f"Transform error for sample {idx}: {e}")
                
                # Close all matplotlib figures to prevent memory leaks
                plt.close('all')
                
                # Clear CUDA cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Continue with next sample
            continue
            
    # Final cleanup before returning
    plt.close('all')
    gc.collect()
    
    return results


def create_clean_image_dataset(data_path, output_dir, class_names, device=None, 
                              colormap='viridis', normalize=True, num_workers=4, 
                              batch_size=16, checkpoint_interval=100, resume=True):
    """
    Create a clean image dataset from the drone RF data with optimizations:
    - Parallel processing
    - Batch processing
    - Memory management
    - Checkpointing
    
    Args:
        data_path: Path to the input data
        output_dir: Directory to save the images
        class_names: List of class names
        device: Torch device to use (CPU or GPU)
        colormap: Matplotlib colormap to use
        normalize: Whether to normalize power spectrum values
        num_workers: Number of parallel workers (threads)
        batch_size: Size of batches to process (per worker)
        checkpoint_interval: How often to save progress
        resume: Whether to resume from previous run
    
    Returns:
        Summary of the created dataset
    """
    # Set device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Initialize multiprocessing method if using CUDA
    if 'cuda' in str(device):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # Create transform
    transform = SpectrogramTransform(device=device)
    
    # Create dataset
    dataset = DroneDataset(path=data_path, device=device, transform=transform)
    
    # Get unique SNR values
    unique_snrs = sorted(set(dataset.get_snrs()))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create structure: class folders only
    for class_idx, class_name in enumerate(class_names):
        (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Load progress if resuming
    start_idx = 0
    if resume and (output_path / "progress.txt").exists():
        with open(output_path / "progress.txt", "r") as f:
            try:
                last_processed = f.read().strip().split(": ")[1]
                start_idx = int(last_processed) + 1
                print(f"Resuming from index {start_idx}")
            except:
                start_idx = 0
    
    # Initialize statistics
    stats = {
        'total_samples': len(dataset),
        'samples_by_class': {class_name: 0 for class_name in class_names},
        'samples_by_snr': {snr: 0 for snr in unique_snrs},
        'processed_samples': 0
    }
    
    # Load existing stats if resuming
    if resume and (output_path / "stats.pt").exists():
        try:
            saved_stats = torch.load(output_path / "stats.pt")
            stats.update(saved_stats)
            print("Loaded existing statistics")
        except:
            pass
    
    # Total samples to process
    total_samples = len(dataset)
    remaining_samples = total_samples - start_idx
    
    if remaining_samples <= 0:
        print("All samples already processed!")
        return stats
    
    print(f"Processing {remaining_samples} samples out of {total_samples} total...")
    
    # Create batches
    all_indices = list(range(start_idx, total_samples))
    batches = [all_indices[i:i + batch_size] for i in range(0, len(all_indices), batch_size)]
    
    # Process batches with progress bar
    with tqdm(total=len(batches)) as pbar:
        for batch_idx, batch in enumerate(batches):
            # Use ThreadPoolExecutor for parallelization
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Split the batch into sub-batches for each worker
                sub_batch_size = max(1, len(batch) // num_workers)
                sub_batches = [batch[i:i + sub_batch_size] for i in range(0, len(batch), sub_batch_size)]
                
                # Process sub-batches in parallel
                future_results = [
                    executor.submit(
                        process_sample_batch, 
                        sub_batch, 
                        dataset, 
                        output_path, 
                        class_names, 
                        colormap, 
                        normalize,
                        transform.n_fft
                    ) 
                    for sub_batch in sub_batches
                ]
                
                # Collect results
                for future in future_results:
                    results = future.result()
                    # Update statistics
                    for target_idx, snr_val in results:
                        class_name = class_names[target_idx]
                        stats['samples_by_class'][class_name] += 1
                        stats['samples_by_snr'][snr_val] += 1
                        stats['processed_samples'] += 1
            
            # Update progress bar
            pbar.update(1)
            
            # Save checkpoint periodically
            if (batch_idx + 1) % checkpoint_interval == 0 or batch_idx == len(batches) - 1:
                last_idx = batch[-1]
                with open(output_path / "progress.txt", "w") as f:
                    f.write(f"Last processed: {last_idx}\n")
                
                # Save statistics
                torch.save(stats, output_path / "stats.pt")
                
                # Clean up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Save final dataset information
    with open(output_path / "dataset_info.txt", "w") as f:
        f.write(f"Total samples: {stats['total_samples']}\n\n")
        
        f.write("Samples by class:\n")
        for class_name, count in stats['samples_by_class'].items():
            f.write(f"  - {class_name}: {count}\n")
        
        f.write("\nSamples by SNR:\n")
        for snr, count in stats['samples_by_snr'].items():
            f.write(f"  - {snr} dB: {count}\n")
    
    print(f"Dataset successfully created at {output_dir}")
    print(f"Processed {stats['processed_samples']} out of {total_samples} samples")
    return stats


if __name__ == "__main__":
    try:
        # Fix matplotlib settings to avoid errors
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Reduce memory usage in matplotlib
        plt.rcParams['figure.max_open_warning'] = 10  # Lower threshold for warning
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        
        # Fix file paths for Kaggle
        print("Current working directory:", os.getcwd())
        
        # Example usage - try to find the data path
        possible_data_paths = [
            "/kaggle/input/noisy-drone-rf-signal-classification-v2/drone_RF_data/",
            "./drone_RF_data/",
            "../input/noisy-drone-rf-signal-classification-v2/drone_RF_data/",
            "../drone_RF_data/"
        ]
        
        # Find the first valid path
        data_path = None
        for path in possible_data_paths:
            if os.path.exists(path):
                data_path = path
                print(f"Found data at: {data_path}")
                break
                
        if data_path is None:
            print("Warning: Could not find data path automatically")
            data_path = "/kaggle/input/noisy-drone-rf-signal-classification-v2/drone_RF_data/"  # Default
            
        # Create output directory
        output_dir = "./clean_spectrograms/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Lower number of workers to avoid overloading
        import multiprocessing
        optimal_workers = 1  # Use just 1 worker to avoid issues
        
        # Read class information (assuming class_stats.csv exists)
        import pandas as pd
        try:
            class_stats_path = os.path.join(data_path, 'class_stats.csv')
            print(f"Looking for class stats at: {class_stats_path}")
            
            if os.path.exists(class_stats_path):
                dataset_stats = pd.read_csv(class_stats_path, index_col=0)
                class_names = dataset_stats['class'].values
                print(f"Found {len(class_names)} classes: {class_names}")
            else:
                print(f"Class stats file not found at {class_stats_path}")
                # Fallback class names if CSV can't be read
                class_names = [f"class_{i}" for i in range(10)]  # Assuming 10 classes
                print(f"Using fallback class names: {class_names}")
        except Exception as e:
            print(f"Error reading class stats: {e}")
            # Fallback class names if CSV can't be read
            class_names = [f"class_{i}" for i in range(10)]  # Assuming 10 classes
            print(f"Using fallback class names due to error: {class_names}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda:0')
        else:
            print("CUDA not available, using CPU")
            device = torch.device('cpu')
            
        # Create clean image dataset with optimizations
        try:
            stats = create_clean_image_dataset(
                data_path=data_path,
                output_dir=output_dir,
                class_names=class_names,
                device=device,
                colormap='viridis',
                normalize=True,
                num_workers=optimal_workers,  # Single thread for stability
                batch_size=4,  # Very small batch size to avoid memory issues
                checkpoint_interval=2,  # Save progress very frequently
                resume=True  # Resume from previous run if available
            )
            
            print("\nDataset creation complete!")
            print(f"Total samples: {stats['total_samples']}")
            print(f"Processed samples: {stats['processed_samples']}")
            print("\nSamples by class:")
            for class_name, count in stats['samples_by_class'].items():
                print(f"  - {class_name}: {count}")
                
        except Exception as e:
            print(f"Error in main processing: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Critical error in main script: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDataset creation complete!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Processed samples: {stats['processed_samples']}")
    print("\nSamples by class:")
    for class_name, count in stats['samples_by_class'].items():
        print(f"  - {class_name}: {count}")