# plot.py
import plotly.graph_objects as go
import pandas as pd

def plot_evaluation_results(df_eval, lookahead, filename='evaluation_plot.html'):
    """
    Generates and saves an interactive HTML plot of the evaluation results with correct alignment.

    Args:
        df_eval (pd.DataFrame): DataFrame containing 'normalized', 'peak', 
                                'prediction', and 'ground_truth_for_prediction' columns.
        lookahead (int): The prediction lookahead window, for labeling.
        filename (str): The name of the output HTML file.
    """
    fig = go.Figure()

    # 1. Plot the "Imminent Peak" ground truth as a shaded background region
    # This shows the windows where the model *should* be predicting a high probability.
    fig.add_trace(go.Scatter(
        x=df_eval.index,
        y=df_eval['ground_truth_for_prediction'],
        mode='lines',
        name=f'Imminent Peak Window (Truth for t+{lookahead})',
        line=dict(width=0), # No line, just the fill
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.2)', # Light orange
        hoverinfo='none' # Hide hover info for this trace
    ))

    # 2. Plot the normalized time series data
    fig.add_trace(go.Scatter(
        x=df_eval.index,
        y=df_eval['normalized'],
        mode='lines',
        name='Normalized Data',
        line=dict(color='royalblue', width=1.5)
    ))

    # 3. Plot the true peaks (actual events) with red markers
    peak_indices = df_eval[df_eval['peak'] == 1].index
    fig.add_trace(go.Scatter(
        x=peak_indices,
        y=df_eval.loc[peak_indices, 'normalized'],
        mode='markers',
        name='Actual Peak Event',
        marker=dict(color='red', size=8, symbol='x-thin', line=dict(width=2))
    ))

    # 4. Plot the model's predicted probability
    # This is correctly aligned with the 'Imminent Peak' ground truth
    fig.add_trace(go.Scatter(
        x=df_eval.index,
        y=df_eval['prediction'],
        mode='lines',
        name='Model Predicted Probability',
        line=dict(color='limegreen', width=2, dash='dash')
    ))

    # Update layout for a clean, professional look
    fig.update_layout(
        title='Model Evaluation: Peak Prediction vs. Aligned Ground Truth',
        xaxis_title='Time Step',
        yaxis_title='Value / Probability',
        yaxis=dict(range=[-0.1, max(1.1, df_eval['normalized'].max()*1.1)]), # Ensure y-axis shows 0-1 range
        legend_title_text='Legend',
        template='plotly_white',
        height=600
    )

    # Save to an HTML file
    fig.write_html(filename)
    print(f"\nInteractive evaluation plot saved to '{filename}'")