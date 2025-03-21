import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .run_test import metric_idx, str_list_to_dict

def get_output_speed(data_metric):
    return data_metric[2]/data_metric[5]

def get_end2end_runtime(data_metric):
    return data_metric[4]+data_metric[5]


def get_new_metrics(dims, data, new_metrics:dict):
    new_dims = dims
    new_dims['metric'] = list(new_metrics.keys())

    new_data =  np.zeros((len(new_dims['context']), len(new_dims['computing_model']), len(new_dims['metric'])))
    for c_i, context_name in enumerate(dims['context']):
        for m_i, model_name in enumerate(dims['computing_model']):
            for metric_i, metric in enumerate(new_dims['metric']):
                remapping = new_metrics[metric]
                if isinstance(remapping, str):
                    new_data[c_i, m_i, metric_i] = data[c_i, m_i, metric_idx[remapping]]
                else:
                    new_data[c_i, m_i, metric_i] = remapping(data[c_i, m_i])
    
    return new_dims, new_data
            

def get_metric(dims, data, remapping):
    new_dims = {'context': dims['context'], 'computing_model': dims['computing_model']}
    new_data =  np.zeros((len(new_dims['context']), len(new_dims['computing_model'])))
    for c_i, context_name in enumerate(new_dims['context']):
        for m_i, model_name in enumerate(new_dims['computing_model']):
            if isinstance(remapping, str):
                new_data[c_i, m_i] = data[c_i, m_i, metric_idx[remapping]]
            else:
                new_data[c_i, m_i] = remapping(data[c_i, m_i])

    return new_dims, new_data
    

def switch_dim_trim_3d(dims, data, order = (0,1,2), keep_idxes = None):
    dim_names = list(dims.keys())
    new_dims = {}
    for i, old_dim_id in enumerate(order):
        dim_name = dim_names[old_dim_id]
        old_dim = dims[dim_name]
        if keep_idxes is not None:
            if keep_idxes[i] is not None:
                old_dim = (np.array(old_dim, dtype=str)[keep_idxes[i]]).tolist()
        new_dims[dim_name] = old_dim
    
    new_data = data
    if keep_idxes is not None:
        if keep_idxes[0] is not None:
            new_data = new_data[keep_idxes[0], :, :]
        if keep_idxes[1] is not None:
            new_data = new_data[:, keep_idxes[1], :]
        if keep_idxes[2] is not None:
            new_data = new_data[:, :, keep_idxes[2]]

    new_data = np.transpose(new_data, order)

    return new_dims, new_data


def tab_2d(dims, data, float_format="%.3f"):

    dim_names = list(dims.keys())
    df = pd.DataFrame(
        data,  # Flatten the first two dimensions
        index=dims[dim_names[0]],
        columns=dims[dim_names[1]]
    )
    
    df = df.T
    print(df.to_latex(float_format=float_format))
    return df



def tab_3d(dims, data, float_format="%.3f"):

    dim_names = list(dims.keys())

    # result = np.swapaxes(result,0,1)
    index = pd.MultiIndex.from_product(
        [ dims[dim_names[0]], dims[dim_names[1]]],
        names=[dim_names[0], dim_names[1]]
    )

    # Create a DataFrame
    df = pd.DataFrame(
        data.reshape(-1, data.shape[2]),  # Flatten the first two dimensions
        index=index,
        columns=dims['metric']
    )
    

    df = df.T
    print(df.to_latex(float_format=float_format))
    return df



if __name__ == '__main__':

    # Define the dimensions
    dims = {
        'context': ['12000', '60000'],
        'computing_model': ['a', 'c', 'd', 'e'],
        'metric': ['model size', 'context length', '#output tokens', 'peak memory', 'initial delay', 'decoding time']
    }

    # Define the 3D array
    result = np.array([
        [
            [14.957543, 12030, 1, 7.500538, 2.3399, 1 / 35.7969],
            [14.957543, 12030, 1, 7.341108, 2.2026, 1 / 35.9064],
            [14.957543, 12030, 1, 7.780913, 2.0874, 1 / 35.8689],
            [14.957543, 12030, 1, 10.219731, 1.9543, 1 / 34.5208]
        ],
        [
            [14.957543, 64030, 1, 22.681978, 15.8177, 1 / 19.8088],
            [14.957543, 64030, 1, 22.472495, 16.99, 1 / 19.6904],
            [14.957543, 64030, 1, 25.530311, 8.5735, 1 / 17.7327],
            [14.957543, 64030, 1, 43, 8.4, 1 / 18]
        ]
    ])

    # result = np.swapaxes(result,0,1)

    # Create a MultiIndex for the rows
    dim_names = list(dims.keys())

    index = pd.MultiIndex.from_product(
        [ dims[dim_names[0]], dims[dim_names[1]]],
        names=[dim_names[0], dim_names[1]]
    )

    # Create a DataFrame
    df = pd.DataFrame(
        result.reshape(-1, result.shape[2]),  # Flatten the first two dimensions
        index=index,
        columns=dims['metric']
    )

    df = df.T


    # Display the DataFrame
    # print(df)
    latex_table = df.to_latex()
    print(latex_table)


    # Create the table
    # table = plt.table(
    #     cellText=df.values,
    #     colLabels=df.columns,
    #     rowLabels=df.index,
    #     loc='center',
    #     cellLoc='center'
    # )

    # # Style the table
    # table.auto_set_font_size(False)
    # table.set_fontsize(12)
    # table.scale(1.2, 1.2)  # Scale the table size

    # plt.show()
