from pylab import *
from sklearn.svm import SVC
from scipy.io import loadmat
from scipy.signal import butter
from scipy.signal import sosfiltfilt
from sklearn.model_selection import train_test_split
from mne_features.feature_extraction import extract_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os
import time
import random
from valid_features import VALID_FEATURES

left_data = []
right_data = []
evaluated_individuals = {}
DATA_DIR = "eeg_data/"
CONVERGENCE_THRESHOLD = 0.00001
_RUN_PREPROCESSING = False
INIT_WITH_SINGULAR = False

ACC_WEIGHT = 0.6
F1_WEIGHT = 0.4

# specific channels for C3 and C4 electrodes
_C3_C4 = [12, 49]  # c3 == 13, c4 == 50
_NUM_OF_CHANNELS = 64
sampl_freq = 512

# desired interval of frequencies
MU_LOWER = 8.0
BETA_HIGHER = 30.0

# constants for evolutionary algorithm
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.6
NUM_GENERATIONS = 10
# NUM_GENERATIONS = 100
POPULATION_SIZE = 30
TOURNAMENT_SIZE = 5
_ALLOW_ELITISM = True

def read_data():
    file_names = sorted(os.listdir(DATA_DIR))
    for file_name in file_names:
        if file_name.endswith(".mat"):

            print(f"Processing {file_name[1:-4]}/52 ...")
            mat_data = loadmat(os.path.join(DATA_DIR, file_name))
            eeg_data = mat_data['eeg']

            sampl_freq = eeg_data['srate'][0][0][0][0]
            print(f"Sampling frequency {sampl_freq} ...")

            eeg_left_data = (eeg_data['movement_left'][0])[0]
            eeg_left_data = eeg_left_data[_C3_C4, :]
            eeg_right_data = (eeg_data['movement_right'][0])[0]
            eeg_right_data = eeg_right_data[_C3_C4, :]

            n_movement_trials = ((eeg_data['n_movement_trials'].item())[0])[0]
            n_total_time_points = eeg_left_data.shape[1]
            n_time_points_per_trial = n_total_time_points // n_movement_trials

            # --- preprocessing of data
            if _RUN_PREPROCESSING:
                print(f"Performing baseline correction...")
                eeg_left_data, eeg_right_data = baseline_correction((eeg_data['rest'][0])[0], eeg_left_data, eeg_right_data)

                print(f"Applying filtering...")
                eeg_left_data, eeg_right_data = butter_filter(eeg_left_data, eeg_right_data)

            print(f"Splitting {file_name} into {n_movement_trials} trials..")
            trial_data_left = eeg_left_data.reshape(2, n_time_points_per_trial, n_movement_trials)
            trial_data_right = eeg_right_data.reshape(2, n_time_points_per_trial, n_movement_trials)

            #  --- transpose to shape for feature extractor (n_epochs, n_channels, n_times_per_trial)
            x_left_transposed = np.transpose(trial_data_left, (2, 0, 1))
            x_right_transposed = np.transpose(trial_data_right, (2, 0, 1))

            left_data.append(x_left_transposed)
            right_data.append(x_right_transposed)


# baseline correction: compute average signal during "rest" period across time axis,
# subtract the noise equivalent to rest state
def baseline_correction(eeg_rest_data, eeg_left_data, eeg_right_data):
    eeg_rest_data = eeg_rest_data[_C3_C4, :]

    rest_avg = np.mean(eeg_rest_data, axis=1, keepdims=True)

    eeg_left_data -= rest_avg
    eeg_right_data -= rest_avg
    return eeg_left_data, eeg_right_data


# filter frequency band relevant for motor movements classification (mu-beta)
def butter_filter(eeg_left_data, eeg_right_data):
    sos = butter(N=4, Wn=[MU_LOWER, BETA_HIGHER], btype='band', fs=float(sampl_freq), output='sos')

    filtered_signal_left = np.zeros_like(eeg_left_data)
    for i in range(eeg_left_data.shape[0]):
        filtered_signal_left[i, :] = sosfiltfilt(sos, eeg_left_data[i, :])

    filtered_signal_right = np.zeros_like(eeg_right_data)
    for i in range(eeg_right_data.shape[0]):
        filtered_signal_right[i, :] = sosfiltfilt(sos, eeg_right_data[i, :])

    return filtered_signal_left, filtered_signal_right


# transform data to features
def data_to_feats(selected_feats):
    all_features = []
    all_labels = []
    for index, data in enumerate(left_data):
        features_left = extract_features(data, sampl_freq, selected_feats)
        features_right = extract_features(right_data[index], sampl_freq, selected_feats)

        all_features.append(np.concatenate((features_left, features_right), axis=0))
        all_labels.append(
            np.concatenate((np.zeros(features_left.shape[0]), np.ones(features_right.shape[0])), axis=0))

    return all_features, all_labels


# map binary vector (individual) onto list of features
def map_to_features(binary_map):
    return [feature for i, feature in enumerate(VALID_FEATURES) if binary_map[i] == 1]


def train_and_eval(features, labels):
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    x_train, x_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    clf = SVC(verbose=True)
    clf.fit(x_train, labels_train)

    labels_pred = clf.predict(x_test)
    accuracy = accuracy_score(labels_test, labels_pred)
    precision = precision_score(labels_test, labels_pred)
    recall = recall_score(labels_test, labels_pred)
    f1 = f1_score(labels_test, labels_pred)
    conf_matrix = confusion_matrix(labels_test, labels_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"Confusion Matrix: \n {conf_matrix}")

    fitness = accuracy * ACC_WEIGHT + f1 * F1_WEIGHT
    print(f"---- Fitness: {fitness}")
    print("------------------")


    return fitness
    # return accuracy


# calculate fitness of given individual
def evaluate_individual(individual):
    print(f"Evaluating individual {individual}.")

    individual_key = tuple(individual)
    if individual_key in evaluated_individuals:
        print(f"Already evaluated this individual, returning cached result: {evaluated_individuals[individual_key]}.")
        return evaluated_individuals[individual_key]

    selected_feats = map_to_features(individual)
    features_intm, labels_intm = data_to_feats(selected_feats)

    # accuracy = fitness of individual
    accuracy = train_and_eval(features_intm, labels_intm)
    evaluated_individuals[individual_key] = accuracy
    return accuracy


# --------------------------------- functions for evolutionary algorithm
def generate_individual():
    return [random.randint(0, 1) for _ in range(len(VALID_FEATURES))]


def generate_population():
    if INIT_WITH_SINGULAR:
        # generate all possible binary vectors with a single 1
        single_one_vectors = [[1 if i == j else 0 for j in range(len(VALID_FEATURES))] for i in
                              range(len(VALID_FEATURES))]

        # randomly generate the rest of the population
        remaining_population_size = POPULATION_SIZE - len(VALID_FEATURES)
        random_population = [generate_individual() for _ in range(remaining_population_size)]

        # Combine the two populations
        population = single_one_vectors + random_population

        return population
    else:
        return [generate_individual() for _ in range(POPULATION_SIZE)]


def get_similarity(individual1, individual2):
    hamming_distance = np.sum(np.fromiter((c1 != c2 for c1, c2 in zip(individual1, individual2)), dtype=int))
    return 1 - (hamming_distance / len(individual1))


def is_too_similar(individual, population, threshold=0.9):
    for existing_individual in population:
        similarity = get_similarity(individual, existing_individual)
        if similarity > threshold:
            return True
    return False


def evolve_population(population):
    new_population = []

    if _ALLOW_ELITISM:
        elites_count = int(0.1 * len(population))
        new_population.extend(population[:elites_count])

        # perform crowding for the remaining population
        remaining_population = population[elites_count:]
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(remaining_population)
            parent2 = tournament_selection(remaining_population)
            print(f"Parent1: {parent1}")
            print(f"Parent2: {parent2}")

            if random.random() < CROSSOVER_RATE:
                offspring = crossover(parent1, parent2)
            else:
                offspring = parent1

            mutate(offspring)

            # check if the offspring is too similar to any individual in the new population
            if not is_too_similar(offspring, new_population):
                new_population.append(offspring)
                print(f"Generated individual {len(new_population)}/{POPULATION_SIZE}")

        return new_population
    else:
        for index, _ in enumerate(population):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            print(f"Parent1: {parent1}")
            print(f"Parent2: {parent2}")

            if random.random() < CROSSOVER_RATE:
                offspring = crossover(parent1, parent2)
            else:
                offspring = parent1

            mutate(offspring)
            new_population.append(offspring)
            print(f"Generated individual {index + 1}/{POPULATION_SIZE}")
        return new_population


def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: evaluate_individual(x), reverse=True)
    return tournament[0]


def crossover(parent1, parent2):
    print(f"Performing crossover..")
    crossover_point = random.randint(0, len(parent1) - 1)
    offspring = parent1[:crossover_point] + parent2[crossover_point:]
    return offspring


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    print(f"Offspring after mutation: {individual}")

# -------------------------------------

def convergence_reached(best_individual, best_individuals):
    if not best_individuals:
        return False
    else:
        difference = abs(best_individual - max(best_individuals))
        return difference <= CONVERGENCE_THRESHOLD


if __name__ == '__main__':
    start_time = time.time()
    read_data()

    best_individuals = []
    population = generate_population()
    for generation in range(NUM_GENERATIONS):
        print(f"\n------------------------------------------------------")
        print(f"-------------------------------------- Generation {generation + 1}/{NUM_GENERATIONS}")
        population = evolve_population(population)

        print(f"Evolving population done. Generated {len(population)} individuals.")
        best_individual = max(population, key=evaluate_individual)

        # if best_individual in best_individuals:
        if False:
            print(f"Reached convergence, stopping after {generation + 1} generations.")
            print(f"Best individual's fitness: {evaluate_individual(best_individual)}, "
                  f"best individual: {best_individual}")
            break
        else:
            best_individuals.append(best_individual)
            print(f"Best individual's fitness: {evaluate_individual(best_individual)}, best individual: {best_individual}")


    end_time = time.time()
    duration = end_time - start_time
    print(f"Total execution time: {duration} seconds.")

    num_indivs = len(evaluated_individuals)
    num_combinations = 2 ** len(VALID_FEATURES)
    print(f"Total number of evaluated individuals: {num_indivs}, out of {num_combinations} possible individuals.")

