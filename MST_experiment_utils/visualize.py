import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from .seaborn_utils import markers, colors
from .run_test import metric_idx, str_list_to_dict

def get_avg_throughput(data, _=None):
    return np.sum(data[:,1]+data[:,2])/np.sum(data[:,4]+data[:,5])

def get_avg_mem_use_ratio(data, ref):
    return np.mean(data[:, 3]/ref[:,3])*100

# def get_avg_mem_use_1d(data, _=None):
#     return data[3]

# def get_context_len_1d(data, _=None):
#     return np.mean(data[3])


def context_plot(data, dims, x_metrix = 'Context Length', y_metrix = 'Peak Memory', title=None, x_title = None, y_title = None, x_lim = None, y_lim = None, size = 200, style = None, save_dir = None):

    model_names = dims['computing_model']

    for m_i, model_name in enumerate(model_names):
        x, y,hue, palette, marker  = [], [], [], [], []
        for c_i, context_name in enumerate(dims['context']):
            x.append(data[c_i, m_i, metric_idx[x_metrix]])
            y.append(data[c_i, m_i, metric_idx[y_metrix]])
            hue.append(model_names[m_i])
            palette.append(colors[m_i])
            marker.append(markers[m_i])

        # print(x)
        sns.scatterplot(
            x=x,  # X-axis value for this point
            y=y,  # X-axis value for this point
            hue=hue,  # Unique identifier for this point
            style=hue,  # Unique marker for this point
            # palette=palette,  # Specific color for this point
            # markers=marker,  # Specific marker for this point
            palette={model_names[m_i]: colors[m_i]},  # Specific color for this point
            markers={model_names[m_i]: markers[m_i]},  # Specific marker for this point
            s=size,  # Size of the points
            legend="full"  # Ensure the legend is displayed
        )
        sns.lineplot(
            x=x,
            y=y,
            color = colors[m_i]
        )

    if x_lim is not None:
        plt.xlim(x_lim)

    if y_lim is not None:
        plt.ylim(y_lim)

    if title is not None:
        plt.title(title)

    plt.legend()


    if save_dir is not None:
        plt.savefig(save_dir)


    # Show the plot
    plt.show()
    plt.close()



def model_scatter(data, dims, x_axis_func, y_axis_func, title=None, x_lim = None, y_lim = None, size = 100, style = None, save_dir = None):
    x, y = [], []
    model_names = dims['computing_model']
    model_idx = str_list_to_dict(model_names)
    
    for i, model_name in enumerate(model_names):
        x.append(x_axis_func(data[:,i,:], data[:,model_idx['vanilla'],:]))
        y.append(y_axis_func(data[:,i,:], data[:,model_idx['vanilla'],:]))
    
    if style is not None:
        sns.set_style(style)


    for i in range(len(x)):
        sns.scatterplot(
            x=[x[i]],  # X-axis value for this point
            y=[y[i]],  # Y-axis value for this point
            hue=[model_names[i]],  # Unique identifier for this point
            style=[model_names[i]],  # Unique marker for this point
            palette={model_names[i]: colors[i]},  # Specific color for this point
            markers={model_names[i]: markers[i]},  # Specific marker for this point
            s=size,  # Size of the points
            legend="full"  # Ensure the legend is displayed
        )

        
    if x_lim is not None:
        plt.xlim(x_lim)

    if y_lim is not None:
        plt.ylim(y_lim)

    if title is not None:
        plt.title(title)

    plt.legend()


    if save_dir is not None:
        plt.savefig(save_dir)

    # Show the plot
    plt.show()
    plt.close()


# def comparison_histogram(dims, data):


if __name__ == '__main__':
    dims = {
        'context': ['12000', '60000'],
        'computing_model':  ['prefill = 512',  # 'prefill = 4096',
                             'prefill = 512, MST', # , 'prefill = 4096, MST', 
                             'MST', 'vanilla'],
        'metric': ['model size', 'context length', '#output tokens', 
                   'peak memory',  'initial delay', 'decoding time']
    }

    result = [[ [14.957543, 12030, 1, 7.500538, 2.3399, 1/35.7969],  
               [14.957543, 12030, 1, 7.341108, 2.2026, 1/35.9064],
               [14.957543, 12030, 1, 7.780913, 2.0874, 1/35.8689],
               [14.957543, 12030, 1, 10.219731, 1.9543, 1/34.5208]],

               [ [14.957543, 64030, 1, 22.681978, 15.8177, 1/19.8088],  
               [14.957543, 64030, 1, 22.472495, 16.99, 1/19.6904],
               [14.957543, 64030, 1, 25.530311, 8.5735, 1/17.7327],
               [14.957543, 64030, 1, 43, 8.4, 1/18]]]
    result = np.array(result)


    context_plot(result, dims)
    # model_scatter(result, dims, get_avg_throughput, get_avg_mem_use_ratio, 'memory use ratio vs. average throughput', x_lim = (0, 2500), y_lim = (40, 100)) #, style='darkgrid')

    '''
    # Input data as NumPy arrays or Python lists
    x = [1, 2, 3, 4, 5]  # X-axis values
    y = [10, 20, 15, 25, 30]  # Y-axis values
    point_ids = ["A", "B", "C", "D", "E"]  # Unique identifier for each point
    colors = ["red", "blue", "green", "purple", "orange"]  # Specific color for each point
    markers = ["o", "s", "D", "^", "X"]  # Specific marker for each point

    # Create a scatter plot
    for i in range(len(x)):
        sns.scatterplot(
            x=[x[i]],  # X-axis value for this point
            y=[y[i]],  # Y-axis value for this point
            hue=[point_ids[i]],  # Unique identifier for this point
            style=[point_ids[i]],  # Unique marker for this point
            palette={point_ids[i]: colors[i]},  # Specific color for this point
            markers={point_ids[i]: markers[i]},  # Specific marker for this point
            s=100,  # Size of the points
            legend="full"  # Ensure the legend is displayed
        )

    # Add a title
    plt.title("Scatter Plot with Specific Colors and Dot Types (NumPy/Lists)")

    # Show the plot
    plt.show()
    '''


'''
# Create example data
data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],  # X-axis values
    "y": [10, 15, 13, 17, 20, 8, 12, 14, 16, 18, 9, 11, 15, 19, 22],  # Y-axis values
    "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"]  # Group labels
})

# Define custom colors for each group
palette = {"A": "red", "B": "blue", "C": "green"}

# Create a scatter plot
scatter = sns.scatterplot(
    x="x",  # X-axis data
    y="y",  # Y-axis data
    hue="group",  # Group by the "group" column
    palette=palette,  # Use custom colors for each group
    s=100,  # Size of the points
    data=data
)

# Optionally connect points within each group
connect_points = True  # Set to False to disable connecting lines
if connect_points:
    sns.lineplot(
        x="x",  # X-axis data
        y="y",  # Y-axis data
        hue="group",  # Group by the "group" column
        palette=palette,  # Use the same custom colors
        dashes=False,  # Disable dashed lines
        markers=False,  # Disable markers on lines
        legend=False,  # Avoid duplicate legend entries
        data=data
    )

# Customize the legend
plt.legend(
    loc="upper right",  # Place the legend in the top-right corner
    bbox_to_anchor=(1.0, 1.0),  # Adjust the exact position of the legend
    title="Groups",  # Add a title to the legend
    fontsize=10,  # Set the font size
    title_fontsize=12  # Set the title font size
)

# Add a title
plt.title("Grouped Scatter Plot with Custom Colors")

# Show the plot
plt.show()
'''

'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example data
groups = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]
colors = ['blue', 'orange', 'green', 'red']
markers = ['o', 's', 'D', '^']  # Different markers for each bar

# Initialize the plot
plt.figure(figsize=(8, 6))

# Plot each bar one at a time
for i, (group, value, color, marker) in enumerate(zip(groups, values, colors, markers)):
    # Plot the bar
    sns.barplot(x=[group], y=[value], color=color, label=group)
    
    # Add a marker on top of the bar
    plt.scatter(i, value, color=color, marker=marker, s=100, label=f'{group} Marker')

# Customize the plot
plt.xlabel('Group')
plt.ylabel('Value')
plt.title('Single Histogram Bars with Markers')
plt.xticks(range(len(groups)), groups)  # Add group labels at the bottom of each bar

# Manage the legend to avoid repetition
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))  # Remove duplicate labels
# plt.legend(unique_labels.values(), unique_labels.keys(), title='Legend')

# Show the plot
plt.show()
'''