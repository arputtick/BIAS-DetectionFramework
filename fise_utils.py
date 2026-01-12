import pandas as pd
from adjustText import adjust_text
import numpy as np
import matplotlib.pyplot as plt
from metric_helper_functions import cosine_similarity, s
from tqdm import tqdm
import os
import time
import json

def fise_experiment(traitlists, language, tests, biases, models, affectstims, 
                    num_traits=None, top_intersectional=5, normalize = False, mean_shift=False):
    '''
    num_traits: Number of traits to randomly sample from the list. If None, load all traits.
    '''
    path = f'results/{language}/FISE/'

    # test_combinations are the combinations of tests and traitlists
    test_combinations = [(test, traitlist) for test in tests for traitlist in traitlists]

    # Conduct all tests and save as json file with format:
    results = {}
    for test, traitlist in test_combinations:
        results[f'{test}_{traitlist}']={}
        for model, embedding in models:
            # Get model name for saving results
            if len(model.split('/')) > 1:
                modelname = model.split('/')[-1]
            else:
                modelname = model

            results[f'{test}_{traitlist}'][model]={}
            # Initialize FISE tester
            fise = FISE(traitlist, language, test, model, 
                        embedding, affectstim = None, bias_axes=biases, num_traits=num_traits)
            category1 = biases[0]
            category2 = biases[1]

            ### Compute word distributions ###
            quadrant_values = fise.compute_quadrant_values(category1, category2, mean_shift=mean_shift)
            distributions = fise.compute_word_distributions(quadrant_values)[1]
            results[f'{test}_{traitlist}'][model]['word_distributions'] = distributions

            ### Compute intersectional traits ###
            top_intersectional_traits = fise.detect_intersectional_traits(quadrant_values, num_traits=top_intersectional, normalize=normalize)
            ### Unnormalized for comparison ###
            if normalize:
                top_intersectional_traits_unorm = fise.detect_intersectional_traits(quadrant_values, num_traits=top_intersectional, normalize=False)
                results[f'{test}_{traitlist}'][model]['intersectional_traits_unorm'] = top_intersectional_traits_unorm
            results[f'{test}_{traitlist}'][model]['intersectional_traits'] = top_intersectional_traits

            ## Plot quadrants ##
            plot = fise.plot_quadrants(quadrant_values)
            # Download plot
            if not os.path.exists(path + f'{test}_{traitlist}'):
                os.makedirs(path + f'{test}_{traitlist}')
            plot.savefig(path + f'{test}_{traitlist}/word_dist_{modelname}.png')
            # Close plot
            plot.close()

            ## Plot top intersectional traits ##
            plot = fise.plot_quadrants(quadrant_values, top_intersectional=top_intersectional, 
                                       labels=True, normalize=normalize)
            plot.savefig(path + f'{test}_{traitlist}/top_int_{modelname}.png')
            plot.close()

            ## Unnormalized for comparison ##
            if normalize:
                plot = fise.plot_quadrants(quadrant_values, top_intersectional=top_intersectional, 
                                       labels=True, normalize=False)
                plot.savefig(path + f'{test}_{traitlist}/top_int_unnorm_{modelname}.png')
                plot.close()

            if affectstims:
                ### Compute affect stim distributions ###
                for affectstim in affectstims:
                    fise.set_affect_stimuli(affectstim)
                    # Compute quadrant affects
                    affect_scores = fise.compute_affect_scores()
                    quadrant_affects = fise.compute_quadrant_affect(quadrant_values, affect_scores, num_traits=5)
                    results[f'{test}_{traitlist}'][model][affectstim] = quadrant_affects

                    plot = fise.plot_quadrant_affects(quadrant_values, affect_scores)
                    # Download plot
                    plot.savefig(path + f'{test}_{traitlist}/{affectstim}_{modelname}.png')
                    # Close plot
                    plot.close()

                    # Plot top intersectional traits
                    plot = fise.plot_quadrant_affects(quadrant_values, affect_scores, 
                                                        labels = True, top_intersectional=top_intersectional, normalize=False)
                    plot.savefig(path + f'{test}_{traitlist}/{affectstim}_top_{modelname}.png')
                    plot.close()



            # Save results as json file
            with open(path + f'{test}_{traitlist}/results.json', 'w') as f:
                # update results file
                # first clear the file
                f.seek(0)
                f.truncate()
                # then write the new results
                json.dump(results, f, indent = 2)
            print(f'Results for {test}_{traitlist}_{model} saved.')

        results = {}

### FISE CLASS ###
class FISE:
    def __init__(self, traitlist, language, test, modelname, 
                 embedding, affectstim, bias_axes, num_traits=None):
        self.traitlist = traitlist
        self.language = language
        self.bias_axes = bias_axes
        self.test = test
        self.embedding = embedding
        self.affectstim = affectstim
        self.FOLDER_PATH = "models/"
        self.model = self.initialize_model(modelname, embedding=embedding)
        try:
            self.affect_model = self.initialize_model(modelname=None,embedding = 'fasttext')
            print('Fasttext model found for affect computation.')
        except:
            print('Fasttext model not found for affect computation. Using the same model as FISE.')
            self.affect_model = self.model
        self.data = self.load_test()
        self.traits = self.get_traits(num_traits)

    def initialize_model(self, modelname, embedding):
        from utils import initialize_model
        model = initialize_model(embedding, self.FOLDER_PATH, modelname, self.language)
        model.loading_model(language=self.language)
        return model
    
    def set_affect_stimuli(self, affectstim):
        self.affectstim = affectstim

    def load_test(self):
        import json
        path = f'datasets/{self.language}/FISE/'
        with open(path + f'{self.test}.json', 'r') as file:
            data = json.load(file)
        return data
    
    def get_traits(self, N=None):
        '''
        Load the trait list from file.
            - N: Number of traits to sample from the list. If None, load all traits.
        '''
        path = f'datasets/{self.language}/FISE/'
        testfile = open(path + f'{self.traitlist}.txt', 'r')
        lines = testfile.read().split('\n')
        traits = []
        for line in lines:
            if line == '':
                continue
            if line[0] == '[':
                # Convert string representation of list to list
                trait = eval(line)
                traits.append(trait)
            else:
                traits.append(line)
        if N:
            np.random.shuffle(traits)
            traits = traits[0:N]
        print('Number of traits:', len(traits))
        return traits
    
    def load_affect_stimuli(self):
        # Load affect stimuli from file
        path = f'datasets/{self.language}/FISE/'
        affect_df = pd.read_csv(path + f'{self.affectstim}.csv', header=None)
        pos_affect = affect_df[0].dropna().tolist()[1:]
        neg_affect = affect_df[1].dropna().tolist()[1:]
        # print(f'Positive affect:', pos_affect)
        # print(f'Negative affect:', neg_affect)
        return pos_affect, neg_affect
    
    def compute_concept_similarity(self, trait, category, concept):
        '''
        trait: str or list of str (for grammatical gender case)
        '''
        concept_words = self.data[category][concept]
        # concept_embeddings = [self.model.get_vector(word) for word in concept_words]
        concept_embeddings = []
        for word in concept_words:
            if type(word) == list:
                word_embedding = np.mean([self.model.get_vector(w) for w in word], axis=0)
            else:
                word_embedding = self.model.get_vector(word)
            concept_embeddings.append(word_embedding)
    
        # If trait is a list of words, average the embeddings
        if type(trait) == list:
            trait_embeddings = [self.model.get_vector(word) for word in trait]
            trait_embedding = np.mean(trait_embeddings, axis=0)
        else:
            trait_embedding = self.model.get_vector(trait)
        similarities = [cosine_similarity(trait_embedding, concept_embedding) for concept_embedding in concept_embeddings]
        similarity = np.mean(similarities)
        return similarity

    def compute_concept_bias(self, trait, category):
        concepts = self.data[category]
        similarities = [self.compute_concept_similarity(trait, category, concept) for concept in concepts]
        difference = similarities[0] - similarities[1]
        return difference

    def compute_quadrant_values(self, category1, category2, mean_shift = False):
        '''
        Parameters:
            category1 (str): The first category. Either 'class', 'race' or 'gender'.
            category2 (str): The second category.
            traits (list): A list of traits to compute intersectional biases of.

        Returns:
            quadrant_values (dict): A dictionary containing the words in each quadrant.
                keys: traits
                values: dict with keys category1 and category2, and values the bias of the train in each category.
        '''
        quadrant_values = {}
        traits = self.traits
        for trait in tqdm(traits, desc='Computing quadrant bias values'):
            bias1 = self.compute_concept_bias(trait, category1)
            bias2 = self.compute_concept_bias(trait, category2)
            quadrant_values[str(trait)] = {category1: bias1,category2: bias2}

        if mean_shift:
            # Shift the mean of the values to 0
            bias1_vals = [values[category1] for values in quadrant_values.values()]
            bias2_vals = [values[category2] for values in quadrant_values.values()]
            mean_bias1 = np.mean(bias1_vals)
            mean_bias2 = np.mean(bias2_vals)
            for trait, values in quadrant_values.items():
                values[category1] -= mean_bias1
                values[category2] -= mean_bias2
        return quadrant_values
    
    def plot_quadrants(self, quadrant_values, show = False, 
                       download = False, top_intersectional=None, labels = False, normalize=False):
        '''
        Plot the words in the quadrants of the plot.
        The x-axis corresponds to category1.
        The y-axis corresponds to category2.
        '''
        if top_intersectional:
            top_intersectional_traits = self.detect_intersectional_traits(quadrant_values, num_traits=top_intersectional, normalize=normalize)
            # Extract words from the top intersectional traits
            traits = []
            for quadrant, trait_list in top_intersectional_traits.items():
                for trait in trait_list:
                    trait = trait[0]
                    if trait[0] == '[':
                        trait = eval(trait)
                    traits.append(trait)

        else:
            traits = self.traits
        # Get the labels for the axes
        category1 = self.bias_axes[0]
        category2 = self.bias_axes[1]
        x_label = category1
        y_label = category2
        x_neg_label = list(self.data[category1].keys())[1]
        x_pos_label = list(self.data[category1].keys())[0]
        y_neg_label = list(self.data[category2].keys())[1]
        y_pos_label = list(self.data[category2].keys())[0]
        x_vals = []
        y_vals = []
        traitlabels = []

        # Set plot limits according to the largest absolute value, plus a small margin
        values = [value for trait, value in quadrant_values.items()]
        x_allvals = [value[x_label] for value in values]
        y_allvals = [value[y_label] for value in values]
        x_max = max([abs(x_val) for x_val in x_allvals])
        y_max = max([abs(y_val) for y_val in y_allvals])
        axis_length_x = x_max + 0.2*x_max
        axis_length_y = y_max + 0.2*y_max
        plt.xlim(-axis_length_x, axis_length_x)
        plt.ylim(-axis_length_y, axis_length_y)
        # max_val = max(max([abs(x_val) for x_val in x_allvals]), max([abs(y_val) for y_val in y_allvals]))
        # axis_length = max_val + 0.2*max_val
        # plt.xlim(-axis_length, axis_length)
        # plt.ylim(-axis_length, axis_length)
            
        # Label the quadrants
        quadrant_1 = x_neg_label + ' ' + y_pos_label
        quadrant_2 = x_pos_label + ' ' + y_pos_label
        quadrant_3 = x_pos_label + ' ' + y_neg_label
        quadrant_4 = x_neg_label + ' ' + y_neg_label
        
        # Place labels in the middle (scaled according to the plot coordinates) of the quadrants in gray font
        # plt.text(-0.5*axis_length, 0.5*axis_length, quadrant_1, color='gray', ha = 'center')
        # plt.text(0.5*axis_length, 0.5*axis_length, quadrant_2, color='gray', ha = 'center')
        # plt.text(0.5*axis_length, -0.5*axis_length, quadrant_3, color='gray', ha = 'center')
        # plt.text(-0.5*axis_length, -0.5*axis_length, quadrant_4, color='gray', ha = 'center')
        plt.text(-0.75*axis_length_x, 0.5*axis_length_y, quadrant_1, color='gray', ha = 'center')
        plt.text(0.75*axis_length_x, 0.5*axis_length_y, quadrant_2, color='gray', ha = 'center')
        plt.text(0.75*axis_length_x, -0.5*axis_length_y, quadrant_3, color='gray', ha = 'center')
        plt.text(-0.75*axis_length_x, -0.5*axis_length_y, quadrant_4, color='gray', ha = 'center')
        for trait, values in quadrant_values.items():
            if trait[0] == '[':
                trait = eval(trait)
            if top_intersectional:
                if trait not in traits:
                    continue
                else:
                    category1 = list(values.keys())[0]
                    category2 = list(values.keys())[1]
                    # print(f'Trait:', trait)
                    # print(category1+":", values[category1])
                    # print(category2+":", str(values[category2])+"\n\n")
                    x_val = values[category1]
                    y_val = values[category2]
                    x_vals.append(x_val)
                    y_vals.append(y_val)
                    if type(trait) == list:
                        # replace the trait pair with e.g. attore/rice or accurato/a
                        if trait[0][-1] == 'o':
                            trait = trait[0]+'/a'
                        elif trait[0][-3:] == 'ore':
                            trait = trait[0]+'/rice'
                        else:
                            trait = trait[0]
                        traitlabels.append(trait)
                    else:
                        traitlabels.append(trait)
            else:
                category1 = list(values.keys())[0]
                category2 = list(values.keys())[1]
                # print(f'Trait:', trait)
                # print(category1+":", values[category1])
                # print(category2+":", str(values[category2])+"\n\n")
                x_val = values[category1]
                y_val = values[category2]
                x_vals.append(x_val)
                y_vals.append(y_val)
                if type(trait) == list:
                # replace the trait pair with e.g. attore/rice or accurato/a
                    if trait[0][-1] == 'o':
                        trait = trait[0]+'/a'
                    elif trait[0][-3:] == 'ore':
                        trait = trait[0]+'/rice'
                    else:
                        trait = trait[0]
                    traitlabels.append(trait)
                else:
                    traitlabels.append(trait)
        if labels:
            # Annotate the points with the corresponding trait
            texts = [plt.text(x_vals[i], y_vals[i], traitlabels[i], ha='center', va='center', fontsize=8) for i in range(len(traitlabels))]
            adjust_text(texts, only_move={'points':'y', 'texts':'y'})

        # Plot the points
        plt.scatter(x_vals, y_vals)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Show x and y axis
        plt.axhline(y=0, color='gray')
        plt.axvline(x=0, color='gray')

        # Axes below the points
        plt.gca().set_axisbelow(True)

        # Don't label axes
        plt.xticks([])
        plt.yticks([])

        # Add title
        plt.title(f'Intersectional plot of {x_label} and {y_label}')

        if download:
            plt.savefig(f'intersectional_plot_{category1}_{category2}.png')

        if show:
            plt.show()

        return plt
    
    def compute_word_distributions(self, quadrant_values):
        '''
        Compute the percentage of words in each quadrant of the plot.
        '''
        quadrant_traits = {}
        # Number of traits
        traits = self.traits
        num_traits = len(traits)

        category1 = self.bias_axes[0]
        category2 = self.bias_axes[1]
        x_pos_label = list(self.data[category1].keys())[0]
        y_pos_label = list(self.data[category2].keys())[0]
        x_neg_label = list(self.data[category1].keys())[1]
        y_neg_label = list(self.data[category2].keys())[1]
        for trait, values in quadrant_values.items():
            category1 = list(values.keys())[0]
            category2 = list(values.keys())[1]
            x_val = values[category1]
            y_val = values[category2]
            if x_val < 0 and y_val > 0:
                quadrant = x_neg_label + ' ' + y_pos_label
            elif x_val > 0 and y_val > 0:
                quadrant = x_pos_label + ' ' + y_pos_label
            elif x_val > 0 and y_val < 0:
                quadrant = x_pos_label + ' ' + y_neg_label
            else:
                quadrant = x_neg_label + ' ' + y_neg_label
            if quadrant in quadrant_traits:
                quadrant_traits[quadrant].append(trait)
            else:
                quadrant_traits[quadrant] = [trait]
        if quadrant not in quadrant_traits:
            quadrant_traits[quadrant] = []
        
        quadrant_percentages = {}
        for quadrant, words in quadrant_traits.items():
            total_words = len(words)
            quadrant_percentages[quadrant] = total_words/num_traits
            # print(f'Quadrant:', quadrant)
            # print(f'Percentage:', quadrant_percentages[quadrant]*100, '%')
            # print(f'Words:', words)
        return quadrant_traits, quadrant_percentages
    
    def detect_intersectional_traits(self, quadrant_values, num_traits=5, normalize=False):
        '''
        For each quadrant, return top num_traits traits (with the largest projection onto the coresponding diagonal).

        NOTE: In the normalized case, it might make sense to resort the top traits by their unnormalized projection.
        '''
        if normalize:
            print('Normalizing vectors for intersectional trait detection.')
        category1 = self.bias_axes[0]
        category2 = self.bias_axes[1]
        traits = self.traits
        x_pos_label = list(self.data[category1].keys())[0]
        y_pos_label = list(self.data[category2].keys())[0]
        x_neg_label = list(self.data[category1].keys())[1]
        y_neg_label = list(self.data[category2].keys())[1]
        intersectional_projections = {}
        unit_vector = np.array([1, 1])/np.sqrt(2)
        anti_unit_vector = np.array([-1, 1])/np.sqrt(2)

        # Identify the traits with the maximum and minimum projection onto the diagonal
        for trait in quadrant_values.keys():
            bias1 = quadrant_values[trait][category1]
            bias2 = quadrant_values[trait][category2]
            vector = np.array([bias1, bias2])
            if normalize:
                vector = vector/np.linalg.norm(vector)
            projection = np.dot(vector, unit_vector)
            anti_projection = np.dot(vector, anti_unit_vector)
            if bias1 < 0 and bias2 > 0:
                quadrant = x_neg_label + ' ' + y_pos_label
            elif bias1 > 0 and bias2 > 0:
                quadrant = x_pos_label + ' ' + y_pos_label
            elif bias1 > 0 and bias2 < 0:
                quadrant = x_pos_label + ' ' + y_neg_label
            else:
                quadrant = x_neg_label + ' ' + y_neg_label
            if quadrant in intersectional_projections:
                intersectional_projections[quadrant].append((trait, projection, anti_projection))
            else:
                intersectional_projections[quadrant] = [(trait, projection, anti_projection)]
        
        top_intersectional_traits = {}
        # For quadrants 1 and 3, sort the traits by their projection onto the anti-diagonal
        # For quadrants 2 and 4, sort the traits by their projection onto the diagonal
        # For quadrants 1 and 2, the top traits have the largest projection
        # For quadrants 3 and 4, the top traits have the smallest projection
        for quadrant, traits in intersectional_projections.items():
            if quadrant == x_neg_label + ' ' + y_pos_label or quadrant == x_pos_label + ' ' + y_neg_label:
                sorted_traits = sorted(traits, key=lambda x: x[2], reverse=True)
                if quadrant == x_pos_label + ' ' + y_neg_label:
                    sorted_traits = sorted_traits[::-1]
                # Remove the projection values but keep anti_projection values
                sorted_traits = [(trait[0], trait[2]) for trait in sorted_traits]
            else:
                sorted_traits = sorted(traits, key=lambda x: x[1], reverse=True)
                if quadrant == x_neg_label + ' ' + y_neg_label:
                    sorted_traits = sorted_traits[::-1]
                # Remove the anti_projection values but keep projection values
                sorted_traits = [(trait[0], trait[1]) for trait in sorted_traits]
            if len(sorted_traits) < num_traits:
                top_intersectional_traits[quadrant] = sorted_traits
            else:
                top_intersectional_traits[quadrant] = sorted_traits[0:num_traits]
            # print(f'Quadrant:', quadrant)
            # print(f'Top intersectional traits:', top_intersectional_traits[quadrant], '\n')

        return top_intersectional_traits
    
    def affect_scorer(self, trait, use_fastext=True):
        '''
        A takes in a trait and outputs an affect score.
        By default, uses the fasttext model in the chosen language.
        Otherwise, uses the same model as FISE.
        '''
        if use_fastext:
            model = self.affect_model
        else:
            model = self.model
        pos, neg = self.load_affect_stimuli()
        if type(trait) == list:
            trait_embedding = np.mean([model.get_vector(word) for word in trait], axis=0)
        else:
            trait_embedding = model.get_vector(trait)
        pos_embeddings = [model.get_vector(word) for word in pos]
        neg_embeddings = [model.get_vector(word) for word in neg]
        affect_score = s(trait_embedding, pos_embeddings, neg_embeddings)
        return affect_score
    
    def compute_affect_scores(self):
        '''
        Compute the affect scores for all traits in the list.
        '''
        affect_scores = {}
        for trait in self.traits:
            affect_scores[str(trait)] = self.affect_scorer(trait)
        return affect_scores

    def compute_quadrant_affect(self, quadrant_values, affect_scores,num_traits=5):
        '''
        For each quadrant, compute the average affect score of the traits in the quadrant.
        - num_traits: number of top affected traits to display
        '''
        # Compute the affect scores for each trait
        category1 = self.bias_axes[0]
        category2 = self.bias_axes[1]
        traits = self.traits
        x_pos_label = list(self.data[category1].keys())[0]
        y_pos_label = list(self.data[category2].keys())[0]
        x_neg_label = list(self.data[category1].keys())[1]
        y_neg_label = list(self.data[category2].keys())[1]
        quadrant_affects = {}
        for trait in quadrant_values.keys():
            bias1 = quadrant_values[trait][category1]
            bias2 = quadrant_values[trait][category2]
            if bias1 < 0 and bias2 > 0:
                quadrant = x_neg_label + ' ' + y_pos_label
            elif bias1 > 0 and bias2 > 0:
                quadrant = x_pos_label + ' ' + y_pos_label
            elif bias1 > 0 and bias2 < 0:
                quadrant = x_pos_label + ' ' + y_neg_label
            else:
                quadrant = x_neg_label + ' ' + y_neg_label
            if quadrant in quadrant_affects:
                quadrant_affects[quadrant].append((trait, affect_scores[str(trait)]))
            else:
                quadrant_affects[quadrant] = [(trait, affect_scores[str(trait)])]

        ### Print num_traits traits with highest and lowest affect scores overall
        traits_scored = quadrant_affects.values()
        # Combine all traits into a single list and sort them by affect score
        all_traits = [trait for traits in traits_scored for trait in traits]
        sorted_traits = sorted(all_traits, key=lambda x: x[1], reverse=True)
        top_traits = sorted_traits[0:num_traits]
        bottom_traits = sorted_traits[-num_traits:]
        # Convert to float to avoid json serialization error
        top_traits = [(trait[0], float(trait[1])) for trait in top_traits]
        bottom_traits = [(trait[0], float(trait[1])) for trait in bottom_traits]
        # Reverse bottom traits to show the most negative first
        bottom_traits = bottom_traits[::-1]
        # print(f'Top {num_traits} traits with highest {self.affectstim} affect scores:',top_traits, '\n')
        # print(f'Top {num_traits} traits with lowest {self.affectstim} affect scores:', bottom_traits, '\n')
        most_affect = {'most_affect':top_traits, 'least_affect':bottom_traits}

        ### Compute average affect score and percentage of positive affect scores for each quadrant
        for quadrant, traits in quadrant_affects.items():
            affect_scores = [trait[1] for trait in traits]
            average_affect = float(np.mean(affect_scores)) # float(), Otherwise can't be saved as json
            percentage_positive = len([score for score in affect_scores if score > 0])/len(affect_scores)
            quadrant_affects[quadrant] = {'average_affect':average_affect, 'percentage_positive':percentage_positive}
            # print(f'Quadrant:', quadrant)
            # print(f'Average affect score:', average_affect, '\n')
            # print(f'Percentage of positive affect scores:', percentage_positive*100, '%\n')
        return quadrant_affects, most_affect

    def plot_quadrant_affects(self, quadrant_values, affect_scores, 
                              show = False, download = False, labels = False, 
                              normalize = False, top_intersectional=None):
        '''
        Modify the plot_quadrants function to color the points based on the affect score.
        '''
        category1 = self.bias_axes[0]
        category2 = self.bias_axes[1]
        if top_intersectional:
            top_intersectional_traits = self.detect_intersectional_traits(quadrant_values, num_traits=top_intersectional, normalize=normalize)
            # Extract words from the top intersectional traits
            traits = []
            for quadrant, trait_list in top_intersectional_traits.items():
                for trait in trait_list:
                    traits.append(trait[0])
        else:
            traits = self.traits
        plt = self.plot_quadrants(quadrant_values, show = show, download= download, 
                                  labels=labels, top_intersectional=top_intersectional)
        # Add color based on affect score
        for trait in traits:
            bias1 = quadrant_values[str(trait)][category1]
            bias2 = quadrant_values[str(trait)][category2]
            affect_score = affect_scores[str(trait)]
            if affect_score > 0:
                color = 'red'
            else:
                color = 'blue'
            plt.scatter(bias1, bias2, color=color)
        
        # Add key for color
        plt.scatter([], [], color='red', label='Positive')
        plt.scatter([], [], color='blue', label='Negative')
        plt.legend()

        # Add title
        plt.title(f'Intersectional plot with {self.affectstim} affect scores')

        if show:
            plt.show()
        return plt