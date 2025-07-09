# evaluate2.py
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from our existing modules
from model import PeakPredictorLSTM
from pmodel_generation import generate_and_prepare_series

def create_ground_truth_for_prediction(peak_series, lookahead):
    """ Creates the "imminent peak" signal for plotting. """
    rolled_max = peak_series.rolling(window=lookahead, min_periods=1).max().shift(-lookahead+1).bfill()
    ground_truth = (rolled_max > 0).astype(int)
    return ground_truth

def prepare_evaluation_dataframe(config):
    """
    Generates a p-model series and prepares a full DataFrame for evaluation.
    """
    print("\nGenerating new evaluation data...")
    df_eval = generate_and_prepare_series(
        length=config['p_model']['series_length_eval'],
        p_value=config['p_model']['p_value'],
        peak_threshold=config['data']['peak_threshold'],
        norm_percentile=config['data']['normalization_percentile']
    )
    df_eval['ground_truth_for_prediction'] = create_ground_truth_for_prediction(
        df_eval['peak'], config['data']['lookahead']
    )
    print(f"Generated a series of length {len(df_eval)} with {df_eval['peak'].sum()} peaks.")
    return df_eval

def plot_evaluation_matplotlib(df, p_value, lookahead=10, filename='data/evaluation_plot_pmodel.png'):
    """
    Generates and saves a PNG plot with styling consistent with the test plotting function.
    Includes step-style shading and shifted prediction curve for anticipating peaks.
    """

    print(f"\nGenerating evaluation plot for P-model (p={p_value})...")
    fig, ax = plt.subplots(figsize=(10, 3.5))

    # --- Setup x-axis data ---
    x_axis_data = df.index
    x_label = "Time-Step"
    signal_label = f"P-model Data (Scaled, p={p_value})"

    # --- Data Normalization for Visualization ---
    plot_df = df.copy()
    max_val = plot_df['normalized'].max()
    scaling_factor = 0.9 / max_val if max_val > 0 else 1.0
    plot_df['normalized_scaled'] = plot_df['normalized'] * scaling_factor

    # 1. Plot the imminent peak warning area
    ax.fill_between(
        x_axis_data, plot_df['ground_truth_for_prediction'], 0,
        color='orange', alpha=0.2, label='Peak Prediction Window',
        step='post'
    )

    # 2. Plot the scaled signal
    ax.plot(
        x_axis_data, plot_df['normalized_scaled'],
        label=signal_label, color='blue', lw=0.6
    )

    # --- Step size for proper shift ---
    if len(x_axis_data) > 1:
        step_size = np.mean(np.diff(x_axis_data))
    else:
        step_size = 1.0

    # 3. Shift prediction to left (anticipating peaks)
    x_pred = x_axis_data - lookahead * step_size
    mask = x_pred >= x_axis_data.min()
    x_pred_masked = x_pred[mask]
    y_pred_masked = plot_df['prediction'].values[mask]

    ax.plot(
        x_pred_masked, y_pred_masked,
        label='Predicted Peak Probability',
        color='limegreen', linestyle='--', lw=1.5
    )

    # 4. Plot the peak markers
    peak_indices = plot_df[plot_df['peak'] == 1].index
    peak_y_values = plot_df.loc[peak_indices, 'normalized_scaled']
    ax.scatter(
        peak_indices, peak_y_values,
        color='red', s=50, edgecolor='black',
        label='Peak Event', zorder=5
    )

    # 5. Title, labels, limits
    ax.set_title(f"LSTM Peak Prediction on P-model Time Series (p = {p_value})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized Value / Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(loc='best', fontsize='small', framealpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Styled evaluation plot saved to '{filename}'")
    plt.show()

def evaluate_and_plot():
    """
    Main function to run the stateful evaluation and generate the styled plot.
    """
    with open('./src/config.json', 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PeakPredictorLSTM(config['model']).to(device)
    try:
        model.load_state_dict(torch.load(config['files']['model_save_path'], map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model file not found at '{config['files']['model_save_path']}'.")
        return

    df_eval = prepare_evaluation_dataframe(config)
    signal = df_eval['normalized'].values
    predictions = []

    print("Running stateful, step-by-step prediction for every time step...")
    hidden_states = model.init_hidden()
    
    with torch.no_grad():
        for t in tqdm(range(len(signal)), desc="Predicting"):
            x_t = signal[t]
            input_tensor = torch.tensor([[[x_t]]], dtype=torch.float32).to(device)
            prediction, new_hidden_states = model.step(input_tensor, hidden_states)
            predictions.append(prediction.item())
            hidden_states = new_hidden_states
            
    df_eval['prediction'] = predictions
    
    plot_evaluation_matplotlib(df_eval, config['p_model']['p_value'])

if __name__ == '__main__':
    evaluate_and_plot()