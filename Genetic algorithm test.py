import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import Rectangle

features = [
        '_closestNuCosmicDist', 'trk_sce_start_x_v', 'trk_sce_start_y_v', 'trk_sce_start_z_v',
        'trk_sce_end_x_v', 'trk_sce_end_y_v', 'trk_sce_end_z_v',
        'reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z',
        'topological_score', 'trk_score_v', 'trk_llr_pid_score_v', 'trk_distance_v',
        'trk_len_v', 'trk_range_muon_mom_v', 'trk_mcs_muon_mom_v'
    ]

# MC
MC_file = '/Users/lukaswystemp/Documents/University/Laboratory Year 3/1. Neutrino Oscillations MicroBoone/true_data/MC_EXT_flattened.pkl'
L = 0.47

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
                (trk_energy_tot < 2))
    
    # Apply selection on dataframe
    frame = frame[selection]
    return frame

# Open file as pandas dataframe
def init_data():

    x_ga = pd.read_pickle(MC_file)
    x_ga = cut_unphysicals(x_ga)


    y_ga = np.where(x_ga['category'] == 21, 1, 0)
    x_ga = x_ga[features]

    training_mask = np.random.rand(x_ga.shape[0]) < 0.8

    x_ga_train = x_ga[training_mask]
    y_ga_train = y_ga[training_mask]

    x_ga_test = x_ga[~training_mask]
    y_ga_test = y_ga[~training_mask]

    return x_ga_train, y_ga_train, x_ga_test, y_ga_test


# Get min and max values for each feature
def define_feature_range(x_ga):
    feature_ranges = {}
    for feature in features:
        # Flatten the lists for vector features to get global min and max
        if feature.endswith('_v'):
            # Flatten the lists
            all_values = x_ga[feature].explode().dropna()
            min_val = all_values.min()
            max_val = all_values.max()
        else:
            min_val = x_ga[feature].min()
            max_val = x_ga[feature].max()
        feature_ranges[feature] = (min_val, max_val)
    return feature_ranges

def generate_conservative_individual(feature_ranges):
    """Generate an individual with initial cuts covering the full range of the data."""
    individual = {}
    for feature in features:
        min_val, max_val = feature_ranges[feature]
        # Start with the full range
        individual[feature] = [min_val, max_val]
    return individual

def generate_initial_population(feature_ranges, population_size):
    """Generate the initial population with conservative cuts."""
    population = []
    for _ in range(population_size):
        individual = generate_conservative_individual(feature_ranges)
        population.append(individual)
    return population

def apply_cuts(individual, x_ga):
    """Apply the selection cuts of an individual to the DataFrame."""
    mask = pd.Series([True] * len(x_ga), index=x_ga.index)
    for feature in features:
        lower_lim, upper_lim = individual[feature]

        feature_mask = (x_ga[feature] >= lower_lim) & (x_ga[feature] <= upper_lim)
        mask = mask & feature_mask
    
    
    y_pred = np.zeros(len(x_ga))
    y_pred[mask.values] = 1  # Events passing cuts are assigned label 1
    x_ga_pred = x_ga[mask]
    print("Number of events passing cuts:", len(y_pred[y_pred == 1]))
    return y_pred, x_ga_pred

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

def fitness(individual, x_ga, y_ga):
    """Compute the fitness of an individual."""
    y_pred, _ = apply_cuts(individual, x_ga)
    efficiency, purity = calculate_eff_purity(y_ga, y_pred)
    # Penalize individuals with zero efficiency or purity
    if efficiency == 0 or purity == 0:
        return 0
    global ALPHA
    ALPHA = 0.9 # importance of purity
    fitness_value = (purity ** ALPHA) * (efficiency ** (1 - ALPHA))#efficiency * purity  # Adjust as needed
    print("Fitness:", fitness_value)
    print("Efficiency:", efficiency)
    print("Purity:", purity)
    return fitness_value

def select_individual(population, fitness_values):
    """Select an individual from the population using roulette wheel selection."""
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        probabilities = [1 / len(population)] * len(population)
    else:
        probabilities = [f / total_fitness for f in fitness_values]
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

def crossover(parent1, parent2):
    """Perform uniform crossover between two parents to produce a child."""
    child = {}
    for feature in features:
        if random.random() < 0.5:
            child[feature] = parent1[feature].copy()
        else:
            child[feature] = parent2[feature].copy()
    return child

def mutate(individual, feature_ranges, mutation_rate=0.1):
    """Mutate an individual's selection cuts."""
    for feature in features:
        if random.random() < mutation_rate:
            min_val, max_val = feature_ranges[feature]
            lower_lim, upper_lim = individual[feature]
            # Adjust limits by a small random fraction of the range
            range_width = max_val - min_val
            delta_lower = random.uniform(-0.1 * range_width, 0.1 * range_width)
            delta_upper = random.uniform(-0.1 * range_width, 0.1 * range_width)
            new_lower_lim = lower_lim + delta_lower
            new_upper_lim = upper_lim + delta_upper
            # Ensure the new limits are within the feature ranges and maintain lower_lim <= upper_lim
            new_lower_lim = max(min_val, min(new_lower_lim, new_upper_lim))
            new_upper_lim = min(max_val, max(new_lower_lim, new_upper_lim))
            individual[feature] = [new_lower_lim, new_upper_lim]
    return individual

def evaluate_population(population, x_ga, y_ga):
    """Evaluate the fitness of each individual in the population."""
    fitness_values = []
    for individual in population:
        fitness_value = fitness(individual, x_ga, y_ga)
        fitness_values.append(fitness_value)
    return fitness_values



def run_genetic_algorithm(x_ga, y_ga, features):
    #Run the genetic algorithm to find the
    # Genetic Algorithm parameters
    population_size = 5
    num_generations = 50 #50
    mutation_rate = 0.1  # Increased mutation rate for faster convergence

    # Initialize population with conservative cuts
    feature_ranges = define_feature_range(x_ga)
    population = generate_initial_population(feature_ranges, population_size)

    best_individual = None
    best_fitness = -1

    for generation in range(num_generations):
        print(f"Generation {generation}")
        fitness_values = evaluate_population(population, x_ga, y_ga)
        
        # Find the best individual
        max_fitness = max(fitness_values)
        max_index = fitness_values.index(max_fitness)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[max_index].copy()
            print(f"New best fitness: {best_fitness}")
        
        new_population = []
        # Elitism: keep the best individual
        new_population.append(best_individual.copy())
        for _ in range(population_size - 1):
            # Select parents
            parent1 = select_individual(population, fitness_values)
            parent2 = select_individual(population, fitness_values)
            # Perform crossover
            child = crossover(parent1, parent2)
            # Mutate
            child = mutate(child, feature_ranges, mutation_rate)
            new_population.append(child)
        population = new_population

    # Output the best individual and its fitness
    print("Best individual:")
    for feature, limits in best_individual.items():
        print(f"{feature}: {limits}")
    print("Best fitness:", best_fitness)
    # Add features and limits to a dictionary
    best_individual_dict = {feature: limits for feature, limits in best_individual.items()}
    print("Best individual dictionary:")
    print(best_individual_dict)

    # Calculate efficiency and purity for the best individual
    y_pred, x_ga_pred = apply_cuts(best_individual, x_ga)
    efficiency, purity = calculate_eff_purity(y_ga, y_pred)
    print("Train Efficiency:", efficiency)
    print("Train Purity:", purity)

    return best_individual


def init_genetic_algorithm():
    x_ga_train, y_ga_train, x_ga_test, y_ga_test = init_data()
    best_individual = run_genetic_algorithm(x_ga_train, y_ga_train, features)


    #Test
    print(f"Genetic algorithm run {i + 1}")
    y_pred, x_ga_pred = apply_cuts(best_individual, x_ga_test)
    test_efficiency, test_purity = calculate_eff_purity(y_ga_test, y_pred)
    print("Test efficiency:", test_efficiency)
    print("Test purity:", test_purity)

    return best_individual, test_purity, test_efficiency










############################################################################################################
# Plot contour plot


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

    plt.text(0.95, 0.05, f'$\\alpha $ = {ALPHA}\n Purity = {test_purity:.3g} \n Efficiency = {test_efficiency:.3g}', transform=plt.gca().transAxes, fontsize=9,
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

sys_uncertainty = 0.1


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
    UNC = 0.15 * mc_counts
    plt.bar(bin_centers, 2*UNC, width = w, bottom = np.array(mc_counts)-UNC, color='grey', alpha=0.7, hatch='//', label = 'Systematic Uncertainty')
    
    # Statistical uncertainty
    STAT = np.sqrt(mc_counts)  # Statistical uncertainty is sqrt of counts
    plt.bar(bin_centers, 2 * STAT, width=w, bottom=np.array(mc_counts) - STAT - UNC, color='pink', alpha=0.7, hatch='//', label='Statistical Uncertainty')


    plt.legend(loc='upper right', labels=[r"$\nu$ NC", r"$\nu_{\mu}$ CC", r"$\nu_e$ CC", r"EXT", r"Out. fid. vol.", r"mis ID",  r"Real Data", r"Systematic Uncertainty", r"Statistical Uncertainty"], prop={'size': 8})
    #plt.text(0.8, 0.8, f"$\Delta m^2 = {delta_m}$ eV$^2$\n$\\sin^2(2\\theta) = {sintheta}$\n$\chi^2 = {chi_squared_val:.2f}$", fontsize = 8, transform=ax.transAxes)
    plt.title("Histogram of the energy spectrum with cut data of $sin^2(2\\theta_{\mu e})$ =" +f"{sintheta}" + "and $\\Delta m_{14}^2$ = "+f"{delta_m}", fontsize = 9)


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






test_purities = []
test_efficiencies = []
contour_3_paths_list = []
for i in range(5):
    best_individual, test_purity, test_efficiency = init_genetic_algorithm()


    print("Now plotting best individual", i + 1)

    data_file = '/Users/lukaswystemp/Documents/University/Laboratory Year 3/1. Neutrino Oscillations MicroBoone/true_data/data_flattened.pkl'
    x_ga_d = pd.read_pickle(data_file)
    x_ga_d = cut_unphysicals(x_ga_d)
    y_ga_d_cut, x_ga_d_cut = apply_cuts(best_individual, x_ga_d)


    x_ga_mc = pd.read_pickle(MC_file)
    x_ga_mc = cut_unphysicals(x_ga_mc)
    y_ga_mc_cut, x_ga_mc_cut = apply_cuts(best_individual, x_ga_mc)


    sintheta_grid_nn, delta_m_grid_nn, chi_squared_vals_nn, min_delta_m_nn, min_index_nn, min_sintheta_nn, min_chi_squared_nn, delta_m_arr_nn = calculate_chi_plot_2(x_ga_mc_cut, x_ga_d_cut)


    theta_lower_lim = 0.001
    m_lower_lim = 0.01

    sintheta_grid_3_1_nn = (1 - np.sqrt(1-0.24))* (1 - np.sqrt(1- sintheta_grid_nn))
    paths = contour_miniboone(sintheta_grid_3_1_nn, delta_m_grid_nn, chi_squared_vals_nn, min_delta_m_nn, min_sintheta_nn, min_chi_squared_nn)

    oscillated_energy_plot(x_ga_mc_cut, x_ga_d_cut, 0.001, 0.1)


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
        plt.plot(*collection.get_paths()[0].vertices.T, lw=1, color = 'blue')
        
        for path in collection.get_paths():
            all_vertices.append(path.vertices)

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
plt.legend(handles= handles, loc = 'upper left')

#plt.scatter(min_sintheta, min_delta_m, color='white', marker='x', label=f'min $\\chi^2$: {min_chi_squared:.2g}')

plt.title('$\\chi^2$ over $(\\Delta m)^2$ and $sin^2(2\\theta)$')
#plt.ylabel('$(\\Delta m)^2$')
#plt.xlabel('$sin^2(2\\theta)$')

test_purity_avg = np.average(test_purities)
test_efficiency_avg = np.average(test_efficiencies)

plt.text(0.95, 0.05, f'$\\alpha $ = {ALPHA}\n Purity = {test_purity_avg:.3g} \n Efficiency = {test_efficiency_avg:.3g}', transform=plt.gca().transAxes, fontsize=9,
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