import json
import pickle

from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pathlib
from sklearn.feature_extraction.text import TfidfVectorizer

from data_reading import read_df


def create_bars_figure(x, y, x_label, y_label, figure_title, full_path_to_save):
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, y, color='blue')
    plt.bar(x, y, color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(figure_title)
    plt.xticks(rotation=70)
    plt.tight_layout()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{round(height, 3)}', ha='center', va='bottom')
    plt.savefig(full_path_to_save)
    plt.close()


def save_results(config, results):
    results_folder_path = pathlib.Path(config['paths']['results_folder_path']) / config['feature_types']
    results_folder_path.mkdir(parents=True, exist_ok=True)

    results_folder_and_file_name = results_folder_path / 'results_per_category.pickle'
    with open(results_folder_and_file_name, 'wb') as pickle_file:
        pickle.dump(results_folder_and_file_name, pickle_file)

    # Extract data
    categories = list(results.keys())
    lengths = [results[category]['len'] for category in categories]
    accuracies = [results[category]['accuracy'] for category in categories]

    # Plot lengths
    create_bars_figure(categories, lengths, 'Category', '# samples', 'Number of Samples per Category',
                       results_folder_path / 'number_of_samples_per_category.png')
    # Plot accuracies
    create_bars_figure(categories, accuracies, 'Category', 'Accuracy', 'Accuracy per Category',
                       results_folder_path / 'accuracies_per_category.png')


def evaluation_per_category(config):
    if config['feature_types'] == 'embeddings':
        df, embedding_feature_names = read_df(config, True)
    else:
        df = read_df(config, False)

    evaluation_per_category_results = {}
    for category in tqdm(df['category'].unique()):
        evaluation_per_category_results[category] = {}
        df_category = df[df['category'] == category]
        # Split data into train and test sets
        if config['feature_types'] == 'embeddings':
            X_train, X_test, y_train, y_test = train_test_split(
                df_category[embedding_feature_names], df_category['one_if_male'], test_size=config['test_size'],
                random_state=config['random_state'])

        elif config['feature_types'] == 'tf_idf':
            tfidf_vect = TfidfVectorizer(ngram_range=tuple(config['tf_idf_params']['ngram_range']))
            X_train, X_test, y_train, y_test = train_test_split(
                df_category['clean_abstract'], df_category['one_if_male'], test_size=config['test_size'],
                random_state=config['random_state'])
            X_train = tfidf_vect.fit_transform(X_train)
            X_test = tfidf_vect.transform(X_test)
        else:
            print('There is no such feature types attribute. Please modify the config file.')

        # Define the undersampling strategy
        undersampler = RandomUnderSampler(random_state=config['random_state'])

        # Resample the training data
        X_train_resampled, y_train_resampled = undersampler.fit_resample(
            X_train, y_train)
        # Resample the test data
        X_test_resampled, y_test_resampled = undersampler.fit_resample(
            X_test, y_test)

        # training
        model = LogisticRegression(max_iter=config['LR_params']['max_iter'], C=config['LR_params']['C'],
                                   penalty=config['LR_params']['penalty'], solver=config['LR_params']['solver'])
        model.fit(X_train_resampled, y_train_resampled)

        # Evaluate on test data
        y_pred = model.predict(X_test_resampled)
        evaluation_per_category_results[category]['len'] = len(y_pred)
        evaluation_per_category_results[category]['accuracy'] = accuracy_score(y_test_resampled, y_pred)

    save_results(config, evaluation_per_category_results)
