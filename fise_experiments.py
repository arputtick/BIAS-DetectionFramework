from fise_utils import fise_experiment
import numpy as np
import os

### EXAMPLE CONFIGURATION ###
# traitlists = ['traitlist', 'occupationlist']
# language = 'en'
# tests = ['FISE_1']
# biases = ['gender', 'race']
# models = [
#           (f"cc.{language}.300.bin",'fasttext'), 
#         #   ('bert-base-uncased','bert_pooling'),
#         #   ('gpt2','gpt_pooling')
#         ]
# affectstims = ['ingressivitystim', 'valencestim']
# num_traits = None # If this is specified, only num_traits traits will be sampled from the trait lists
# top_intersectional = 15 # If num_traits is specified, this is the number of top intersectional traits to be used
# normalize = True # If True, the normalized top intersectional traits will be used

#### EXPERIMENT 1 ####
# All tests on fasttext model

## EXPERIMENT CONFIGURATION ###
traitlists = ['traitlist_gender_bal_gg', 'occupationlist_gender_bal_gg']
language = 'it'
tests = ['FISE_1', 'FISE_IT1', 'FISE_IT2']
biases = ['gender', 'race']
models = [
            (f"cc.{language}.300.bin",'fasttext'), 
            # ('dbmdz/bert-base-italian-uncased','bert_pooling'),
            # ('dbmdz/bert-base-italian-xxl-uncased','bert_pooling'),
            # ('GroNLP/gpt2-small-italian','gpt_pooling')
         ]
affectstims = ['ingressivitystim_gender_bal', 'valencestim_gender_bal']
num_traits = None
top_intersectional = 15
normalize = True

### EXPERIMENT ###
fise_experiment(traitlists, language, tests, biases, models, affectstims, 
                top_intersectional=top_intersectional, num_traits=num_traits, normalize=normalize)

### MOVE RESULTS TO EXPERIMENT FOLDER ###
results_path = f'results/{language}/FISE/'
experiment_path = results_path + 'Experiment_1/'
# Create experiment_path if it does not exist
if not os.path.exists(experiment_path):
    os.system(f"mkdir {experiment_path}")
# Move all files from results_path to experiment_path.
# Identify experiment folders to exclude. Any folder with 'Experiment' in its name.
experiment_folders = [f for f in os.listdir(results_path) if 'Experiment' in f]
# Move all files except experiment_folder from results_path to experiment_path.
for f in os.listdir(results_path):
    if f not in experiment_folders:
        os.system(f"mv {results_path}{f} {experiment_path}")

#### EXPERIMENT 2 ####
# Testing occupation bias across all models

### EXPERIMENT CONFIGURATION ###
traitlists = ['occupationlist_gender_bal']
language = 'it'
tests = ['FISE_IT1', 'FISE_IT2']
biases = ['gender', 'race']
models = [
            (f"cc.{language}.300.bin",'fasttext'), 
            ('dbmdz/bert-base-italian-uncased','bert_pooling'),
            # ('dbmdz/bert-base-italian-xxl-uncased','bert_pooling'),
            ('GroNLP/gpt2-small-italian','gpt_pooling')
         ]
affectstims = ['ingressivitystim_gender_bal', 'valencestim_gender_bal']
num_traits = None
top_intersectional = 15
normalize = True

### EXPERIMENT ###
fise_experiment(traitlists, language, tests, biases, models, affectstims, 
                top_intersectional=top_intersectional, num_traits=num_traits, normalize=normalize)

### MOVE RESULTS TO EXPERIMENT FOLDER ###
results_path = f'results/{language}/FISE/'
experiment_path = results_path + 'Experiment_2/'
# Create experiment_path if it does not exist
if not os.path.exists(experiment_path):
    os.system(f"mkdir {experiment_path}")
# Identify experiment folders to exclude. Any folder with 'Experiment' in its name.
experiment_folders = [f for f in os.listdir(results_path) if 'Experiment' in f]
# Move all files except experiment_folder from results_path to experiment_path.
for f in os.listdir(results_path):
    if f not in experiment_folders:
        os.system(f"mv {results_path}{f} {experiment_path}")

#### EXPERIMENT 3 ####
# Testing occupational bias with grammatical gender for fasttext and bert

### EXPERIMENT CONFIGURATION ###
traitlists = ['occupationlist', 'occupationlist_gender_bal', 'occupationlist_gender_bal_gg']
language = 'it'
tests = ['FISE_IT1']
biases = ['gender', 'race']
models = [
            (f"cc.{language}.300.bin",'fasttext'), 
            ('dbmdz/bert-base-italian-uncased','bert_pooling'),
         ]
affectstims = ['ingressivitystim_gender_bal', 'valencestim_gender_bal']
num_traits = None
top_intersectional = 15
normalize = True

### EXPERIMENT ###
fise_experiment(traitlists, language, tests, biases, models, affectstims, 
                top_intersectional=top_intersectional, num_traits=num_traits, normalize=normalize)

### MOVE RESULTS TO EXPERIMENT FOLDER ###
results_path = f'results/{language}/FISE/'
experiment_path = results_path + 'Experiment_3/'
# Create experiment_path if it does not exist
if not os.path.exists(experiment_path):
    os.system(f"mkdir {experiment_path}")
# Identify experiment folders to exclude. Any folder with 'Experiment' in its name.
experiment_folders = [f for f in os.listdir(results_path) if 'Experiment' in f]
# Move all files except experiment_folder from results_path to experiment_path.
for f in os.listdir(results_path):
    if f not in experiment_folders:
        os.system(f"mv {results_path}{f} {experiment_path}")


#### EXPERIMENT 4 ####
# Testing effect of gender balancing on affective stimuli

### EXPERIMENT CONFIGURATION ###
traitlists = ['traitlist_gender_bal_gg', 'occupationlist_gender_bal_gg']
language = 'it'
tests = ['FISE_IT1']
biases = ['gender','race']
models = [
            (f"cc.{language}.300.bin",'fasttext'), 
         ]
affectstims = ['ingressivitystim', 'ingressivitystim_gender_bal', 'valencestim', 'valencestim_gender_bal']
num_traits = None
top_intersectional = 15
normalize = True

### EXPERIMENT ###
fise_experiment(traitlists, language, tests, biases, models, affectstims, 
                top_intersectional=top_intersectional, num_traits=num_traits, normalize=normalize)

### MOVE RESULTS TO EXPERIMENT FOLDER ###
results_path = f'results/{language}/FISE/'
experiment_path = results_path + 'Experiment_4/'
# Create experiment_path if it does not exist
if not os.path.exists(experiment_path):
    os.system(f"mkdir {experiment_path}")
# Identify experiment folders to exclude. Any folder with 'Experiment' in its name.
experiment_folders = [f for f in os.listdir(results_path) if 'Experiment' in f]
# Move all files except experiment_folder from results_path to experiment_path.
for f in os.listdir(results_path):
    if f not in experiment_folders:
        os.system(f"mv {results_path}{f} {experiment_path}")

#### EXPERIMENT 5 ####
# Testing effect of FISE_IT1 vs FISE_IT1_GG

### EXPERIMENT CONFIGURATION ###
traitlists = ['occupationlist_gender_bal_gg']
language = 'it'
tests = ['FISE_IT1', 'FISE_IT1_GG']
biases = ['gender','race']
models = [
            (f"cc.{language}.300.bin",'fasttext'),
            ('dbmdz/bert-base-italian-uncased','bert_pooling'),
         ]
affectstims = ['ingressivitystim_gender_bal', 'valencestim_gender_bal']
num_traits = None
top_intersectional = 15
normalize = True

### EXPERIMENT ###
fise_experiment(traitlists, language, tests, biases, models, affectstims, 
                top_intersectional=top_intersectional, num_traits=num_traits, normalize=normalize)

### MOVE RESULTS TO EXPERIMENT FOLDER ###
results_path = f'results/{language}/FISE/'
experiment_path = results_path + 'Experiment_5/'
# Create experiment_path if it does not exist
if not os.path.exists(experiment_path):
    os.system(f"mkdir {experiment_path}")
# Identify experiment folders to exclude. Any folder with 'Experiment' in its name.
experiment_folders = [f for f in os.listdir(results_path) if 'Experiment' in f]
# Move all files except experiment_folder from results_path to experiment_path.
for f in os.listdir(results_path):
    if f not in experiment_folders:
        os.system(f"mv {results_path}{f} {experiment_path}")

#### EXPERIMENT 6 ####
# Testing the affect of mean_shift

### EXPERIMENT CONFIGURATION ###
traitlists = ['occupationlist_gender_bal_gg']
language = 'it'
tests = ['FISE_IT1']
biases = ['gender','race']
models = [
            (f"cc.{language}.300.bin",'fasttext'),
            ('dbmdz/bert-base-italian-uncased','bert_pooling'),
         ]
affectstims = ['ingressivitystim_gender_bal', 'valencestim_gender_bal']
num_traits = None
top_intersectional = 15
normalize = True
mean_shift = True

### EXPERIMENT ###
fise_experiment(traitlists, language, tests, biases, models, affectstims, 
                top_intersectional=top_intersectional, num_traits=num_traits, normalize=normalize, mean_shift=mean_shift)

### MOVE RESULTS TO EXPERIMENT FOLDER ###
results_path = f'results/{language}/FISE/'
experiment_path = results_path + 'Experiment_6/'
# Create experiment_path if it does not exist
if not os.path.exists(experiment_path):
    os.system(f"mkdir {experiment_path}")
# Identify experiment folders to exclude. Any folder with 'Experiment' in its name.
experiment_folders = [f for f in os.listdir(results_path) if 'Experiment' in f]
# Move all files except experiment_folder from results_path to experiment_path.
for f in os.listdir(results_path):
    if f not in experiment_folders:
        os.system(f"mv {results_path}{f} {experiment_path}")