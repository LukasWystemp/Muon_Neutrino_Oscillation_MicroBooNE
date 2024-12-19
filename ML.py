import numpy as np
import uproot3
import pickle

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import Neutrino_functions

from math import *
import scipy as sci

# MACHINE LEARNING IMPORTS
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import initializers
from sklearn.utils import class_weight



# Data
def cut_unphysicals(frame):
    
    # Basic variables present in dataframe 
    trk_start_x_v = frame['trk_sce_start_x_v']        # cm
    trk_start_y_v = frame['trk_sce_start_y_v']        # cm
    trk_start_z_v = frame['trk_sce_start_z_v']        # cm
    trk_end_x_v = frame['trk_sce_end_x_v']            # cm
    trk_end_y_v = frame['trk_sce_end_y_v']            # cm
    trk_end_z_v = frame['trk_sce_end_z_v']            # cm
    reco_x = frame['reco_nu_vtx_sce_x']               # cm
    reco_y = frame['reco_nu_vtx_sce_y']               # cm
    reco_z = frame['reco_nu_vtx_sce_z']               # cm
    topological = frame['topological_score']          # N/A
    trk_score_v = frame['trk_score_v']                # N/A
    trk_dis_v = frame['trk_distance_v']               # cm
    trk_len_v = frame['trk_len_v']                    # cm
    trk_energy_tot = frame['trk_energy_tot']          # GeV 
    
    
    
    # select the conditions you want to apply, here is an initial condition to get you started.
    selection = ((trk_len_v > 0) &
                (trk_energy_tot < 1.5))
    
    # Apply selection on dataframe
    frame = frame[selection]
    return frame


# MC
MC_file = './data/MC_EXT_flattened.pkl'

# Data
data_file = './data/data_flattened.pkl'

threshold = 0.5


features = [
    '_closestNuCosmicDist', 'trk_sce_start_x_v', 'trk_sce_start_y_v', 'trk_sce_start_z_v',
    'trk_sce_end_x_v', 'trk_sce_end_y_v', 'trk_sce_end_z_v',
    'reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z',
    'topological_score', 'trk_score_v', 'trk_llr_pid_score_v', 'trk_distance_v',
    'trk_len_v', 'trk_range_muon_mom_v', 'trk_mcs_muon_mom_v'
]


def normalise(X):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X


#min_vals = X.min(axis=0)  # Minimum value for each column
#max_vals = X.max(axis=0)
#range_vals = max_vals - min_vals
#Normalized_X = (X - min_vals) / range_vals


#x_norm = unnorm(x_norm)



def define_model(X_train):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),

        Dense(64, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),

        Dense(23, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])
    return model


def initial_run():
    # Open file as pandas dataframe
    MC_EXT = pd.read_pickle(MC_file)

    # removing 'Subevent' from data
    MC_EXT = MC_EXT.drop('Subevent', axis = 1)


    df = MC_EXT.copy(deep = True)
    df = cut_unphysicals(df)
    y = np.where(df['category'] == 21, 1, 0)

    X = df[features]


    X = np.array(X, dtype = np.float64)
    y = np.array(y)

    print("no of signal: ", len(y[y == 1]))
    print("no of background: ", len(y[y == 0]))
    x_norm = normalise(X)

    X_train, X_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2, random_state=42)

    model = define_model(X_train)

    # weights
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train) # class  weight uses label encoded y_train not one hot encoded y_train
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))


    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights_dict)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")

    y_pred_prob = model.predict(X_test).flatten()
    threshold = 0.5
    y_pred = (y_pred_prob >= threshold).astype(int)


    plot_confusion_matrix(y_test, y_pred)


    test_efficiency, test_purity = calculate_eff_purity(y_test, y_pred)
    print("Initial Run Test efficiency", test_efficiency)
    print("Initial Run Test purity", test_purity)
    print("--------------------")



    #investigate_model_specific_energy_range(df, model, threshold)



def investigate_model_specific_energy_range(df, model, threshold):
        # Investigate model at specific energy range

    df_mc_energy_range = cut_test_energy_range(df)
    y_mc_energy_range = np.where(df_mc_energy_range['category'] == 21, 1, 0)
    mc_energy_range = df_mc_energy_range[features]
    mc_energy_range = np.array(mc_energy_range, dtype = np.float64)
    y_mc_energy_range_pred = model.predict(mc_energy_range).flatten()
    y_mc_energy_range_pred = (y_mc_energy_range_pred >= threshold).astype(int)
    _, energy_range_accuracy = model.evaluate(mc_energy_range, y_mc_energy_range)
    print(f"Accuracy on energy range: {energy_range_accuracy}")
    print("Kept percentage of mc data: ", len(y_mc_energy_range_pred[y_mc_energy_range_pred == 1]) )
    print("total amount of data", len(y_mc_energy_range_pred))


    x_data_energy_range = pd.read_pickle(data_file)
    x_data_energy_range = cut_unphysicals(x_data_energy_range)
    x_d_energy_range = cut_test_energy_range(x_data_energy_range)
    data_energy_range = x_d_energy_range[features]
    data_energy_range = np.array(data_energy_range, dtype = np.float64)
    y_d_energy_range_pred = model.predict(data_energy_range).flatten()
    y_d_energy_range = (y_d_energy_range_pred >= threshold).astype(int)
    print("Kept percentage of data: ", len(y_d_energy_range_pred[y_d_energy_range_pred == 1]))
    print("total amount of data", len(y_d_energy_range_pred))



def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Signal'])
    disp.plot()
    plt.show()

    print(y_test)
    print(y_pred)
    print(len(y_pred))
    print(len(y_pred[y_pred == 0]))
    print(len(y_pred[y_pred == 1]))

def calculate_eff_purity(y_test, y_pred): #y_test, y_pred
    """Calculate efficiency and purity based on true labels and predictions."""
    no_events_surviving_selection = np.sum(y_pred == 1)
    no_of_signal_events_that_pass_selection = np.sum(y_pred == 1)
    total_number_of_events = len(y_test)
    efficiency = no_of_signal_events_that_pass_selection / total_number_of_events
    # Handle division by zero
    if no_events_surviving_selection == 0:
        purity = 0
    else:
        y_cut = y_test[y_pred == 1]
        no_of_signal_events_that_pass_selection = np.sum(y_cut == 1)
        purity = no_of_signal_events_that_pass_selection / no_events_surviving_selection

    #original_signal_count = len(y_original[y_original == 1])
    #original_event_count = len(y_original)

    #cut_signal_count = len(y_cut[y_cut == 1])
    #cut_event_count = len(y_cut)

    # efficiency  = signal events that pass the selection / all original events
    #efficiency = cut_signal_count / original_event_count


    #purity = cut_signal_count / cut_event_count

    return efficiency, purity


def cut_test_energy_range(frame):
    
    # Basic variables present in dataframe 
    trk_start_x_v = frame['trk_sce_start_x_v']        # cm
    trk_start_y_v = frame['trk_sce_start_y_v']        # cm
    trk_start_z_v = frame['trk_sce_start_z_v']        # cm
    trk_end_x_v = frame['trk_sce_end_x_v']            # cm
    trk_end_y_v = frame['trk_sce_end_y_v']            # cm
    trk_end_z_v = frame['trk_sce_end_z_v']            # cm
    reco_x = frame['reco_nu_vtx_sce_x']               # cm
    reco_y = frame['reco_nu_vtx_sce_y']               # cm
    reco_z = frame['reco_nu_vtx_sce_z']               # cm
    topological = frame['topological_score']          # N/A
    trk_score_v = frame['trk_score_v']                # N/A
    trk_dis_v = frame['trk_distance_v']               # cm
    trk_len_v = frame['trk_len_v']                    # cm
    trk_energy_tot = frame['trk_energy_tot']          # GeV 
    
    
    
    # select the conditions you want to apply, here is an initial condition to get you started.
    selection = ( (trk_energy_tot < 0.8) & (trk_energy_tot > 0.6))
    
    # Apply selection on dataframe
    frame = frame[selection]
    return frame


def init_ml():
    # Predict on full data
    # MC data
    print("Now do cuts on data")
    df_mc = pd.read_pickle(MC_file)
    df_mc = cut_unphysicals(df_mc)

    y_mc = np.where(df_mc['category'] == 21, 1, 0)


    mc = df_mc[features]


    mc = np.array(mc, dtype = np.float64)
    y_mc = np.array(y_mc)

    print("no of signal: ", len(y_mc[y_mc == 1]))
    print("no of background: ", len(y_mc[y_mc == 0]))


    mc_norm = normalise(mc)

    model = define_model(mc_norm)




    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_mc), y = y_mc) # class  weight uses label encoded y_train not one hot encoded y_train
    class_weights_dict = dict(zip(np.unique(y_mc), class_weights))


    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(mc_norm, y_mc, epochs=1, batch_size=32, validation_data=(mc_norm, y_mc), class_weight=class_weights_dict)

    # Evaluate the model
    loss, accuracy = model.evaluate(mc_norm, y_mc)
    print(f"Test Accuracy: {accuracy:.2f}")






    y_mc_pred = model.predict(mc_norm).flatten()
    y_mc_pred = (y_mc_pred >= threshold).astype(int)

    print(y_mc_pred)

    test_efficiency, test_purity = calculate_eff_purity(y_mc, y_mc_pred)

    print("--------------------")
    print("Test Efficiency: ", test_efficiency)
    print("Test Purity: ", test_purity)

    mc_cut = df_mc[y_mc_pred == 1]


    # Real data
    df_data = pd.read_pickle(data_file)
    df_data = cut_unphysicals(df_data)
    data = df_data[features]
    data = np.array(data, dtype = np.float64)

    data_norm = normalise(data)
    y_d_pred = model.predict(data_norm).flatten()
    y_d_pred = (y_d_pred >= threshold).astype(int)

    data_cut = df_data[y_d_pred == 1]

    return mc_cut, data_cut, test_purity, test_efficiency



############################################################################################################
# Plot contour plot

L = 0.47
def contour_miniboone(sintheta_grid, delta_m_grid, chi_squared_vals, min_delta_m, min_sintheta, min_chi_squared):
    # Load data
    LSND_data = pd.read_csv('./data/DataSet_LSND.csv').to_numpy()
    MiniBooNE_data = pd.read_csv('./data/DataSet_MiniBooNE.csv').to_numpy()

    # Plot data
    plt.plot(LSND_data[:,0],LSND_data[:,1],'o')
    plt.plot(MiniBooNE_data[:,0],MiniBooNE_data[:,1],'o')

    # Producing MiniBooNE/LSND legend
    LSND_path = mpatches.Patch(color='tab:blue', label = 'LSND')
    MINI_path = mpatches.Patch(color='tab:orange', label = 'MiniBooNE')
    first_legend = plt.legend(handles=[LSND_path, MINI_path], loc = 'lower left', fontsize = 12)
    plt.gca().add_artist(first_legend)

    plt.contourf(sintheta_grid, delta_m_grid, chi_squared_vals, levels=40, cmap = 'magma')
    plt.colorbar()

    contour_1 = plt.contour(sintheta_grid, delta_m_grid, chi_squared_vals, levels=[31], colors = "white") # 68
    contour_2 = plt.contour(sintheta_grid, delta_m_grid, chi_squared_vals, levels=[37], colors = "green") # 90
    contour_3 = plt.contour(sintheta_grid, delta_m_grid, chi_squared_vals, levels=[41], colors = "blue") # 95
    contour_4 = plt.contour(sintheta_grid, delta_m_grid, chi_squared_vals, levels=[48.3], colors = "yellow") # 99

    handles = [
            plt.Line2D([0], [0], color='white', label='68%'),
            plt.Line2D([0], [0], color='green', label='90%'),
            plt.Line2D([0], [0], color='blue', label='95%'),
            plt.Line2D([0], [0], color='yellow', label='99%')
        ]
    plt.legend(handles= handles, loc = 'upper left')

    #plt.scatter(min_sintheta, min_delta_m, color='white', marker='x', label=f'min $\\chi^2$: {min_chi_squared:.2g}')

    plt.title('$\\chi^2$ over $(\\Delta m)^2$ and $sin^2(2\\theta)$')
    #plt.ylabel('$(\\Delta m)^2$')
    #plt.xlabel('$sin^2(2\\theta)$')

    plt.text(0.95, 0.05, f'Purity = {test_purity:.3g} \n Efficiency = {test_efficiency:.3g}', transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))


    plt.xlabel(r'$sin^2(2\theta_{\mu e})=sin^2(\theta_{24})sin^2(2\theta_{14})$',fontsize=20)
    plt.ylabel(r'$\Delta$ $m_{14}^2$',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale('log')
    plt.xscale('log')

    plt.xlim(0.0001,np.max(sintheta_grid))

    plt.show()

    return contour_3


def prob_func(amplitude, delta_m, E):
    return 1 - amplitude * (np.sin(1.27 * delta_m * L / E)**2)

sys_uncertainty = 0.15


def chi_squared(bin_counts, exp_counts):
    return np.sum((bin_counts - exp_counts)**2 / ((sys_uncertainty * bin_counts) ** 2 + bin_counts))

def call_chi(mc_counts, mb_counts, mc_bin_edges):
    bin_centers = 0.5 * (mc_bin_edges[:-1] + mc_bin_edges[1:])
    chi = np.zeros(len(mc_counts))
    for idx in range(len(bin_centers)):
        chi[idx] = chi_squared(mc_counts[idx], mb_counts[idx])
    return np.sum(chi)


custom_palette = {
    4: sns.color_palette("bright")[0],  # mis ID
    5: sns.color_palette("bright")[1],  # Out Fid. Vol.
    7: sns.color_palette("bright")[2],  # EXT
    10: sns.color_palette("bright")[3],  # nu_e CC
    21: sns.color_palette("bright")[4],  # nu_mu CC
    31: sns.color_palette("bright")[5]   # nu NC
}


def oscillated_energy_plot(mc_data, true_data, sintheta, delta_m):
    fig, ax = plt.subplots(1, 1)

    weight = mc_data["weight"] * prob_func(sintheta, delta_m, mc_data["true_E"])

    mc_counts, mc_bin_edges = np.histogram(mc_data["trk_energy_tot"], bins=20, weights=weight)
    mb_counts, _ = np.histogram(true_data['trk_energy_tot'], bins=mc_bin_edges)


    bin_centers = 0.5 * (mc_bin_edges[:-1] + mc_bin_edges[1:])
    chi_squared_val = call_chi(mc_counts, mb_counts, mc_bin_edges)
    print("chi squared: ", chi_squared_val)


    
    ax = sns.histplot(data=mc_data, x="trk_energy_tot", multiple="stack", hue="category", weights=weight, bins=20, legend=False, palette='deep')


    #hist_err(ax, mc_counts, mc_bin_edges)
    ax.errorbar(bin_centers, mb_counts, xerr= 0.5*(mc_bin_edges[1] - mc_bin_edges[0]), fmt='o', color='black', label='Data', markersize=3, capsize=0, elinewidth=1)

    # Systematic uncertainty
    w = mc_bin_edges[1] - mc_bin_edges[0]
    UNC = 0.1 * mc_counts
    plt.bar(bin_centers, 2*UNC, width = w, bottom = np.array(mc_counts)-UNC, color='grey', alpha=0.7, hatch='//', label = 'Systematic Uncertainty')
    
    # Statistical uncertainty
    STAT = np.sqrt(mc_counts)  # Statistical uncertainty is sqrt of counts
    plt.bar(bin_centers, 2 * STAT, width=w, bottom=np.array(mc_counts) - STAT - UNC, color='pink', alpha=0.7, hatch='//', label='Statistical Uncertainty')


    plt.legend(loc='upper right', labels=[r"$\nu$ NC", r"$\nu_{\mu}$ CC", r"$\nu_e$ CC", r"EXT", r"Out. fid. vol.", r"mis ID",  r"Real Data", r"Systematic Uncertainty", r"Statistical Uncertainty"], prop={'size': 8})
    #plt.text(0.8, 0.8, f"$\Delta m^2 = {delta_m}$ eV$^2$\n$\\sin^2(2\\theta) = {sintheta}$\n$\chi^2 = {chi_squared_val:.2f}$", fontsize = 8, transform=ax.transAxes)
    plt.title("Histogram of the energy spectrum with cut data of $sin^2(2\\theta_{\mu e})$ =" +f"{sintheta}" + "and $\\Delta m_{14}^2$ = "+f"{delta_m}", fontsize = 9)


# statistical fluctuations due to bins at high energy
# model is much better at classifying mc data than real data
# compare event reconstruction at that energy range
# shape changes significantly



    plt.xlabel('Reconstructed Track Energy [GeV]', fontsize=10)
    plt.ylabel('Event Counts', fontsize=10)
    plt.show()

def calculate_chi_plot_2(mc_data, real_data):
    real_bin_counts, real_bin_edges = np.histogram(real_data["trk_energy_tot"], bins=20)
    exp_bin_centers = 0.5 * (real_bin_edges[:-1] + real_bin_edges[1:])

    dim = 60

    delta_m_arr = np.logspace(np.log10(0.01), np.log10(100), dim)
    sintheta_arr = np.logspace(np.log10(0.001), np.log10(1), dim)

    delta_m_grid, sintheta_grid = np.meshgrid(delta_m_arr, sintheta_arr, indexing='ij')
    chi_squared_vals = np.zeros(delta_m_grid.shape)



    mu_indices = mc_data["category"] == 21
    non_mu_indices = ~mu_indices

    all_weights_original = mc_data["weight"].copy()
    all_trk_energy_tot_original = mc_data["trk_energy_tot"].copy()
    all_true_E_original = mc_data["true_E"].copy()



    #mc_data_mu = mc_data[mc_data['category'] == 21]
    #mc_data_non_mu = mc_data[mc_data['category'] != 21]
        
    #weight_mu = mc_data_mu["weight"]
    #weight_non_mu = mc_data_non_mu["weight"]

    for i in range(dim):
        for j in range(dim):

            weights = all_weights_original.copy()
            probs = prob_func(sintheta_grid[i, j], delta_m_grid[i, j], all_true_E_original)

            weights.loc[mu_indices] *= probs


            bin_counts, _ = np.histogram(mc_data["trk_energy_tot"], bins = 20, weights = weights)
            chi_squared_vals[i, j] = chi_squared(bin_counts, real_bin_counts)
            #print("m", sintheta_grid[i, j], "theta", delta_m_grid[i, j])
            #print(chi_squared_vals[i, j])

    min_index = np.unravel_index(np.argmin(chi_squared_vals), chi_squared_vals.shape)

    min_delta_m = delta_m_grid[min_index]
    min_sintheta = sintheta_grid[min_index]

    min_chi_squared = chi_squared_vals[min_index]

    print(f"Minimum chi-squared value: {min_chi_squared}")
    print(f"Corresponding delta_m value: {min_delta_m}")
    print(f"Corresponding sintheta value: {min_sintheta}")
    return sintheta_grid, delta_m_grid, chi_squared_vals, min_delta_m, min_index, min_sintheta, min_chi_squared, delta_m_arr




initial_run()

"""
mc_cut, data_cut, test_purity, test_efficiency = init_ml()



sintheta_grid, delta_m_grid, chi_squared_vals, min_delta_m, min_index, min_sintheta, min_chi_squared, delta_m_arr = calculate_chi_plot_2(mc_cut, data_cut)

sintheta_grid_3_1_nn = (1 - np.sqrt(1-0.24))* (1 - np.sqrt(1- sintheta_grid))
paths = contour_miniboone(sintheta_grid_3_1_nn, delta_m_grid, chi_squared_vals, min_delta_m, min_sintheta, min_chi_squared)

oscillated_energy_plot(mc_cut, data_cut, 0.001, 0.1)

"""




test_purities = []
test_efficiencies = []
contour_3_paths_list = []
for i in range(5):

    mc_cut, data_cut, test_purity, test_efficiency = init_ml()

    print("Now plotting best individual", i + 1)



    sintheta_grid_nn, delta_m_grid_nn, chi_squared_vals_nn, min_delta_m_nn, min_index_nn, min_sintheta_nn, min_chi_squared_nn, delta_m_arr_nn = calculate_chi_plot_2(mc_cut, data_cut)


    theta_lower_lim = 0.001
    m_lower_lim = 0.01

    sintheta_grid_3_1_nn = (1 - np.sqrt(1-0.24))* (1 - np.sqrt(1- sintheta_grid_nn))
    paths = contour_miniboone(sintheta_grid_3_1_nn, delta_m_grid_nn, chi_squared_vals_nn, min_delta_m_nn, min_sintheta_nn, min_chi_squared_nn)

    oscillated_energy_plot(mc_cut, data_cut, 10e-3, 10e1)


    print("Best individuals range")
    contour_3_paths_list.append(paths)
    test_purities.append(test_purity)
    test_efficiencies.append(test_efficiency)







LSND_data = pd.read_csv('./data/DataSet_LSND.csv').to_numpy()
MiniBooNE_data = pd.read_csv('./data/DataSet_MiniBooNE.csv').to_numpy()
 
# Plot data
plt.plot(LSND_data[:,0],LSND_data[:,1],'o')
plt.plot(MiniBooNE_data[:,0],MiniBooNE_data[:,1],'o')

# Producing MiniBooNE/LSND legend
LSND_path = mpatches.Patch(color='tab:blue', label = 'LSND', zorder = 1)
MINI_path = mpatches.Patch(color='tab:orange', label = 'MiniBooNE', zorder = 2)
first_legend = plt.legend(handles=[LSND_path, MINI_path], loc = 'lower left', fontsize = 12)
plt.gca().add_artist(first_legend)


plt.contourf(sintheta_grid_3_1_nn, delta_m_grid_nn, chi_squared_vals_nn, levels=40, cmap = 'magma', zorder = 0)

all_vertices = []
for contour in contour_3_paths_list:
    for collection in contour.collections:
        # Extract the paths and plot them manually

        try:
            plt.plot(*collection.get_paths()[0].vertices.T, lw=1, color = 'blue')
            
            for path in collection.get_paths():
                all_vertices.append(path.vertices)
        except IndexError:
            continue

leftmost_contour = min(all_vertices, key=lambda vertices: np.min(vertices[:, 0]))
#rightmost_contour = max(all_vertices, key=lambda vertices: np.max(vertices[:, 0]))
rightmost_contour = None
max_x_value = -np.inf

# Loop through each array
for array in all_vertices:
    # Find the index of the row with y closest to 1
    idx = np.argmin(np.abs(array[:, 1] - 1))
    x_value = array[idx, 0]  # Get the corresponding x-value
    
    # Update the rightmost contour if this x-value is greater than the current maximum
    if x_value > max_x_value:
        max_x_value = x_value
        rightmost_contour = array

# Sort the vertices to define the fill region
left_x, left_y = leftmost_contour[:, 0], leftmost_contour[:, 1]
right_x, right_y = rightmost_contour[:, 0], rightmost_contour[:, 1]

print("Leftmost contour x-range:", np.min(leftmost_contour[:, 0]), np.max(leftmost_contour[:, 0]))
print("Rightmost contour x-range:", np.min(rightmost_contour[:, 0]), np.max(rightmost_contour[:, 0]))

# Combine the vertices to create a closed polygon
combined_x = np.concatenate([left_x, right_x[::-1]])
combined_y = np.concatenate([left_y, right_y[::-1]])
plt.plot(combined_x, combined_y, color='red', lw=2, label='Shaded Area Outline')


# Create a filled polygon
plt.fill(combined_x, combined_y, color='yellow', alpha = 0.5, label='95% Confidence Belt', zorder = 4)

#contour_1 = plt.contour(sintheta_grid_3_1_nn, delta_m_grid_nn, chi_squared_vals_nn, levels=[31], colors = "white") # 68
#contour_2 = plt.contour(sintheta_grid_3_1_nn, delta_m_grid_nn, chi_squared_vals_nn, levels=[37], colors = "green") # 90
#contour_4 = plt.contour(sintheta_grid_3_1_nn, delta_m_grid_nn, chi_squared_vals_nn, levels=[48.3], colors = "yellow") # 99
    
plt.colorbar()

handles = [
            plt.Line2D([0], [0], color='white', label='68%'),
            plt.Line2D([0], [0], color='green', label='90%'),
            plt.Line2D([0], [0], color='blue', label='95%'),
            plt.Line2D([0], [0], color='yellow', label='99%')
        ]


handles.append(plt.Line2D([0], [0], color='yellow', alpha=0.5, lw=2, label='95% Confidence Belt'))
plt.legend(handles=handles, loc='upper left')

#plt.scatter(min_sintheta, min_delta_m, color='white', marker='x', label=f'min $\\chi^2$: {min_chi_squared:.2g}')

plt.title('$\\chi^2$ over $(\\Delta m)^2$ and $sin^2(2\\theta)$')
#plt.ylabel('$(\\Delta m)^2$')
#plt.xlabel('$sin^2(2\\theta)$')

test_purity_avg = np.average(test_purities)
test_efficiency_avg = np.average(test_efficiencies)

plt.text(0.95, 0.05, f'Purity = {test_purity_avg:.3g} \n Efficiency = {test_efficiency_avg:.3g}', transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))


plt.xlabel(r'$sin^2(2\theta_{\mu e})=sin^2(\theta_{24})sin^2(2\theta_{14})$',fontsize=20)
plt.ylabel(r'$\Delta$ $m_{14}^2$',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.yscale('log')
plt.xscale('log')

plt.xlim(0.0001,np.max(sintheta_grid_3_1_nn))

plt.show()
