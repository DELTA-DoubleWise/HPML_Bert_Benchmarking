import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.lines import Line2D

task = "classification"

tasks = ["classification", "qa", "mlm", "token_classification"]

for task in tasks:

    # Load the data from both CSV files
    data_flash = pd.read_csv(task + '/log_Bert_flash_attention_2_' + task +'.csv')
    data_sdpa = pd.read_csv(task + '/log_Bert_sdpa_' + task +'.csv')

    # Ensure the columns are read correctly
    print(data_flash.columns)
    print(data_sdpa.columns)

    # Convert the appropriate columns to categorical types for better plotting
    categories = {
        ' batch_size': [1, 2, 4, 8, 16, 32, 64, 128],
        ' seq_len': [128, 256, 512],
        ' pad percentage': [0, 0.1, 0.2, 0.5, 0.75]
    }
    for column, cats in categories.items():
        data_flash[column] = pd.Categorical(data_flash[column], categories=cats, ordered=True)
        data_sdpa[column] = pd.Categorical(data_sdpa[column], categories=cats, ordered=True)

    # Define a color palette
    palette = sns.color_palette("husl", len(categories[' pad percentage']))

    # List to keep track of filenames for combining later
    file_paths = []

    # Variables to store min and max values for each metric
    speedup_min = float('inf')
    speedup_max = float('-inf')
    mem_saved_min = float('inf')
    mem_saved_max = float('-inf')

    # Find min and max values for each metric across both methods
    for seq_len in [128, 256, 512]:
        for metric in [' Speedup (%)', ' Mem saved (%)']:
            metric_min = min(data_flash[data_flash[' seq_len'] == seq_len][metric].min(), data_sdpa[data_sdpa[' seq_len'] == seq_len][metric].min())
            metric_max = max(data_flash[data_flash[' seq_len'] == seq_len][metric].max(), data_sdpa[data_sdpa[' seq_len'] == seq_len][metric].max())
            if 'Speedup' in metric:
                speedup_min = min(speedup_min, metric_min)
                speedup_max = max(speedup_max, metric_max)
            elif 'Mem saved' in metric:
                mem_saved_min = min(mem_saved_min, metric_min)
                mem_saved_max = max(mem_saved_max, metric_max)

    # Create and save plots with both methods included
    for i, seq_len in enumerate([128, 256, 512]):
        for j, metric in enumerate([' Speedup (%)', ' Mem saved (%)']):
            plt.figure(figsize=(6, 4))
            ax = plt.gca()
            # Plot for flash_attention method
            sns.lineplot(data=data_flash[data_flash[' seq_len'] == seq_len], x=' batch_size', y=metric, hue=' pad percentage', style=' pad percentage', markers=True, dashes=False, palette=palette, ax=ax, legend=False)
            # Plot for sdpa_memory_efficient method
            sns.lineplot(data=data_sdpa[data_sdpa[' seq_len'] == seq_len], x=' batch_size', y=metric, hue=' pad percentage', style=' pad percentage', markers=True, dashes=[(2,2)], palette=palette, ax=ax, legend=False)
            ax.set_xscale('log', base=2)  # Set the x-axis to a logarithmic scale with base 2
            ax.set_title(f'Seq Len: {seq_len} - {metric}')
            ax.set_xlabel('Batch Size (log scale)')
            ax.set_ylabel(metric)

            # Set consistent y-axis limits for each metric
            if 'Speedup' in metric:
                ax.set_ylim(speedup_min, speedup_max)
            elif 'Mem saved' in metric:
                ax.set_ylim(mem_saved_min, mem_saved_max)

            # Create custom legends
            custom_lines = [Line2D([0], [0], color=palette[n], lw=4, linestyle='-') for n in range(len(categories[' pad percentage']))]  # Solid lines for flash_attention
            custom_lines_dotted = [Line2D([0], [0], color=palette[n], lw=4, linestyle=':') for n in range(len(categories[' pad percentage']))]  # Dotted lines for sdpa_memory_efficient
            ax.legend(custom_lines + custom_lines_dotted, [f'Flash - {p}' for p in categories[' pad percentage']] + [f'SDPA - {p}' for p in categories[' pad percentage']], title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            file_name = task + f'/SeqLen_{seq_len}_{metric.strip().replace(" ", "_")}.png'
            plt.savefig(file_name)
            file_paths.append(file_name)
            plt.close()

    # Combine images into a single image
    images = [Image.open(x) for x in file_paths]
    widths, heights = zip(*(i.size for i in images))

    # Assuming all images are the same size
    total_width = 2 * max(widths)
    total_height = 3 * max(heights)

    # Create new image to place others onto
    new_im = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    count = 0
    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.width
        count += 1
        if count % 2 == 0:
            x_offset = 0
            y_offset += im.height

    new_im.save(task + '/combined_plot.png')  # Save the combined image
