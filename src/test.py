# test2.py
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from our existing modules
from model import PeakPredictorLSTM

def load_and_prepare_csv(file_path, peak_threshold, lookahead):
    """
    Loads a CSV, normalizes it, and creates both peak labels and the
    "imminent peak" ground truth signal.
    """
    print(f"\n--- Processing file: {os.path.basename(file_path)} ---")
    try:
        # Assuming header: Time-step,Amplitude
        df = pd.read_csv(file_path)
        df.columns = ['time', 'amplitude']
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return None

    # Normalize the signal
    amp = df['amplitude'].values
    if amp.min() >= 0 and amp.max() <= 1:
        df['normalized_amplitude'] = df['amplitude']
    else:
        min_val, max_val = amp.min(), amp.max()
        df['normalized_amplitude'] = (amp - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(amp)

    # Create ground truth peak labels
    df['peak'] = (df['normalized_amplitude'] > peak_threshold).astype(int)
    
    # Create the "imminent peak" ground truth
    rolled_max = df['peak'].rolling(window=lookahead, min_periods=1).max().shift(-lookahead+1).bfill()
    df['ground_truth_for_prediction'] = (rolled_max > 0).astype(int)
            
    return df

def plot_test_matplotlib(df, file_path, lookahead):
    """
    Generates and saves a PNG plot with highly customized labels and styles
    based on the input filename.
    The prediction curve is shifted left by 'lookahead' steps (considering step size)
    to show the prediction at the time it was made (anticipating the event).
    """

    print(f"Generating customized plot for {os.path.basename(file_path)}...")
    fig, ax = plt.subplots(figsize=(10, 3.5))

    # --- File-Specific Plotting Logic ---
    filename_base = os.path.basename(file_path)
    if 'heat_flux_iter.csv' in filename_base:
        x_axis_data = df['time'] * 0.05
        x_label = "Timestep (ms)"
        signal_label = "Input Signal (ITER)"
        output_filename = "./data/heat_flux_prediction.png"
    elif 'sdo-A10A-29nv20.csv' in filename_base:
        x_axis_data = df['time']
        x_label = "Time-Step"
        signal_label = "Input Signal (SDO)"
        output_filename = "./data/sdo_prediction.png"
    else:  # Fallback for other files
        x_axis_data = df.index
        x_label = "Time-Step"
        signal_label = "Input Signal"
        output_filename = "./data/unknown_signal_prediction.png"

    # --- Data Normalization for Visualization ---
    plot_df = df.copy()
    max_val = plot_df['normalized_amplitude'].max()
    scaling_factor = 0.9 / max_val if max_val > 0 else 1.0
    plot_df['normalized_scaled'] = plot_df['normalized_amplitude'] * scaling_factor

    # 1. Plot the imminent peak warning area
    ax.fill_between(
        x_axis_data, plot_df['ground_truth_for_prediction'], 0,
        color='orange', alpha=0.2, label='Peak Prediction Window',
        step='post' 
    )

    # 2. Plot the SCALED signal data
    ax.plot(
        x_axis_data, plot_df['normalized_scaled'],
        label=signal_label, color='blue', lw=0.6
    )

    # --- Calculate average step size for correct shifting ---
    if len(x_axis_data) > 1:
        step_size = np.mean(np.diff(x_axis_data))
    else:
        step_size = 1.0  # fallback if only one point

    # 3. Prepare shifted prediction for plotting (shift left by lookahead * step_size)
    x_pred = x_axis_data - lookahead * step_size

    # Mask to avoid plotting out-of-bound values (negative or less than min time)
    mask = x_pred >= x_axis_data.min()
    x_pred_masked = x_pred[mask]
    y_pred_masked = plot_df['prediction'].values[mask]

    # Plot shifted prediction curve
    ax.plot(
        x_pred_masked, y_pred_masked,
        label=f'Peak Prediction Probability',
        color='limegreen', linestyle='--', lw=1.5
    )

    # 4. Plot the ground truth peak markers
    peak_indices = plot_df[plot_df['peak'] == 1].index
    peak_x_values = x_axis_data[peak_indices]
    peak_y_values = plot_df.loc[peak_indices, 'normalized_scaled']
    ax.scatter(
        peak_x_values, peak_y_values,
        color='red', s=50, edgecolor='black',
        label='Peak Event', zorder=5
    )

    # 5. Set titles, labels, and y-axis limits
    if 'sdo-A10A-29nv20.csv' in filename_base:
        ax.set_title("Max10-Averaged AIA 304 Å Series (SDO) with Predicted Peak Probability")
        ax.set_ylabel("SDO AIA 304 Å – A10A (Normalized) / Probability")
    elif 'heat_flux_iter.csv' in filename_base:
        ax.set_title("Simulated Heat Flux for SPARC-to-ITER Scenario with Predicted Peak Probability")
        ax.set_ylabel("Normalized Heat Flux (GW/m²) / Probability")
    else:
        ax.set_title("Peak Prediction on Test Signal")
        ax.set_ylabel("Normalized Value / Probability")

    ax.set_xlabel(x_label)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(loc='best', fontsize='small', framealpha=0.7)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Customized plot saved to '{output_filename}'")
    plt.show()

def test_and_plot():
    """
    Main function to run stateful inference on specific test files and generate plots.
    """
    try:
        with open('src/config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found.")
        return

    peak_threshold = config['data']['peak_threshold']
    lookahead = config['data']['lookahead']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PeakPredictorLSTM(config['model']).to(device)
    try:
        model.load_state_dict(torch.load(config['files']['model_save_path'], map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file not found at '{config['files']['model_save_path']}'. Please run train.py first.")
        return

    # --- Find and Process Specific Test Files ---
    test_files_to_find = ['sdo-A10A-29nv20.csv', 'heat_flux_iter.csv']
    test_dir = 'test'

    for filename in test_files_to_find:
        file_path = os.path.join(test_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: Test file not found at '{file_path}'. Skipping.")
            continue
        
        df = load_and_prepare_csv(file_path, peak_threshold, lookahead)
        if df is None:
            continue
        
        signal = df['normalized_amplitude'].values
        predictions = []

        print("Running stateful, step-by-step prediction...")
        hidden_states = model.init_hidden()
        
        with torch.no_grad():
            for t in tqdm(range(len(signal)), desc=f"Predicting on {filename}"):
                x_t = signal[t]
                input_tensor = torch.tensor([[[x_t]]], dtype=torch.float32).to(device)
                prediction, new_hidden_states = model.step(input_tensor, hidden_states)
                predictions.append(prediction.item())
                hidden_states = new_hidden_states
        
        df['prediction'] = predictions
        plot_test_matplotlib(df, file_path, lookahead)

if __name__ == '__main__':
    # Create a dummy test directory and files for demonstration if they don't exist
    if not os.path.exists('./test'):
        print("Creating dummy './test' directory and files for demonstration.")
        os.makedirs('../test')
        # Dummy SDO-like data
        time_sdo = np.arange(300)
        amp_sdo = 150 + np.sin(time_sdo / 30) * 20 + np.random.rand(300) * 5
        amp_sdo[100:105] += 40
        pd.DataFrame({'Time-step': time_sdo, 'Amplitude': amp_sdo}).to_csv('./test/sdo-A10A-29nv20.csv', index=False)
        # Dummy ITER-like data
        time_iter = np.arange(500)
        amp_iter = 5 + np.sin(time_iter / 50) * 2
        amp_iter[200:203] += 5
        amp_iter[400:402] += 6
        pd.DataFrame({'Time-step': time_iter, 'Amplitude': amp_iter}).to_csv('./test/heat_flux_iter.csv', index=False)

    test_and_plot()