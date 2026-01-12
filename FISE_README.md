# FISE Experiments README

## Overview

The Flexible Intersectional Stereotype Extraction (FISE) experiments are designed to measure bias in language models by analyzing how traits and occupations are distributed across different demographic groups. This framework evaluates intersectional bias by examining word embeddings in relation to gender, race, and other demographic attributes. Tests are available for both English and Italian languages, with specific adaptations for grammatical gender in Italian.

## Prerequisites

### Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `fasttext==0.9.3` - For FastText embeddings
- `transformers==4.42.3` - For BERT and GPT models
- `torch==2.3.1` - For PyTorch-based models
- `pandas==2.2.2` - For data handling
- `matplotlib==3.9.1` - For visualization
- `tqdm==4.66.4` - For progress bars

### Model Setup

Ensure you have the required language models available:

**FastText Models:**
- Place FastText models in the `models/` directory
- For Italian: `cc.it.300.bin`
- For English: `cc.en.300.bin`

**Transformer Models:**
- BERT models are downloaded automatically from Hugging Face
- Common models: `bert-base-uncased`, `dbmdz/bert-base-italian-uncased`
- GPT models: `gpt2`, `GroNLP/gpt2-small-italian`

## Available Test Datasets

### English (en)
- **FISE_1**: Main English bias evaluation dataset
- Located in: `datasets/en/FISE/`

### Italian (it)
- **FISE_1**: Standard FISE test adapted for Italian
- **FISE_IT1**: Italian-specific test variant 1
- **FISE_IT2**: Italian-specific test variant 2
- **FISE_IT1_GG**: Italian test with grammatical gender considerations
- Located in: `datasets/it/FISE/`

## Trait and Stimulus Lists

### Trait Lists
- **traitlist**: Standard personality traits
- **occupationlist**: Standard occupations
- **traitlist_gender_bal_gg**: Gender-balanced traits with grammatical gender
- **occupationlist_gender_bal_gg**: Gender-balanced occupations with grammatical gender

### Affective Stimuli
- **ingressivitystim**: Aggressiveness-related stimuli
- **valencestim**: Valence (positive/negative) stimuli
- **ingressivitystim_gender_bal**: Gender-balanced aggressiveness stimuli
- **valencestim_gender_bal**: Gender-balanced valence stimuli

## Running FISE Experiments

### Basic Configuration

Edit the configuration section in `fise_experiments.py`:

```python
### EXPERIMENT CONFIGURATION ###
traitlists = ['traitlist_gender_bal_gg', 'occupationlist_gender_bal_gg']
language = 'it'  # or 'en'
tests = ['FISE_1', 'FISE_IT1', 'FISE_IT2']
biases = ['gender', 'race']
models = [
    (f"cc.{language}.300.bin", 'fasttext'),
    ('dbmdz/bert-base-italian-uncased', 'bert_pooling'),
    ('GroNLP/gpt2-small-italian', 'gpt_pooling')
]
affectstims = ['ingressivitystim_gender_bal', 'valencestim_gender_bal']
num_traits = None  # If specified, sample only this many traits
top_intersectional = 15  # Number of top intersectional traits to analyze
normalize = True  # Whether to use normalized intersectional traits
mean_shift = False  # Whether to apply mean shift correction
```

### Configuration Parameters

- **traitlists**: List of trait/occupation files to use
- **language**: Target language ('en', 'it')
- **tests**: FISE test datasets to run
- **biases**: Bias dimensions to evaluate (e.g., 'gender', 'race')
- **models**: List of (model_name, embedding_type) tuples
- **affectstims**: Affective stimulus files to use
- **num_traits**: Limit number of traits (None = use all)
- **top_intersectional**: Number of top intersectional traits to report
- **normalize**: Apply normalization to intersectional trait detection
- **mean_shift**: Apply mean shift correction to embeddings

### Model Types

**FastText:**
```python
(f"cc.{language}.300.bin", 'fasttext')
```

**BERT:**
```python
('bert-base-uncased', 'bert_pooling')
('dbmdz/bert-base-italian-uncased', 'bert_pooling')
```

**GPT:**
```python
('gpt2', 'gpt_pooling')
('GroNLP/gpt2-small-italian', 'gpt_pooling')
```

### Running Experiments

1. **Configure your experiment** in `fise_experiments.py`
2. **Run the experiment:**
   ```bash
   python fise_experiments.py
   ```

### Example Experiment Configurations

#### Experiment 1: Comprehensive FastText Analysis
```python
traitlists = ['traitlist_gender_bal_gg', 'occupationlist_gender_bal_gg']
language = 'it'
tests = ['FISE_1', 'FISE_IT1', 'FISE_IT2']
biases = ['gender', 'race']
models = [(f"cc.{language}.300.bin", 'fasttext')]
affectstims = ['ingressivitystim_gender_bal', 'valencestim_gender_bal']
```

#### Experiment 2: Cross-Model Comparison
```python
traitlists = ['occupationlist_gender_bal']
language = 'it'
tests = ['FISE_IT1', 'FISE_IT2']
models = [
    (f"cc.{language}.300.bin", 'fasttext'),
    ('dbmdz/bert-base-italian-uncased', 'bert_pooling'),
    ('GroNLP/gpt2-small-italian', 'gpt_pooling')
]
```

#### Experiment 3: Grammatical Gender Impact
```python
traitlists = ['occupationlist', 'occupationlist_gender_bal', 'occupationlist_gender_bal_gg']
tests = ['FISE_IT1']
models = [
    (f"cc.{language}.300.bin", 'fasttext'),
    ('dbmdz/bert-base-italian-uncased', 'bert_pooling')
]
```

## Results

### Output Location
Results are saved to: `results/{language}/FISE/`

### Automatic Organization
Each experiment automatically:
1. Creates a numbered experiment folder (e.g., `Experiment_1/`)
2. Moves results from the main results directory to the experiment folder
3. Preserves existing experiment folders

### Result Format
Results are saved as JSON files containing:
- **word_distributions**: Distribution of words across bias quadrants
- **intersectional_traits**: Top intersectional traits detected
- **intersectional_traits_unorm**: Unnormalized intersectional traits (if normalize=True)

## Pre-configured Experiments

The `fise_experiments.py` file includes six pre-configured experiments:

1. **Experiment 1**: Comprehensive FastText analysis across all tests
2. **Experiment 2**: Cross-model comparison on occupation bias
3. **Experiment 3**: Grammatical gender impact analysis
4. **Experiment 4**: Effect of gender balancing on affective stimuli
5. **Experiment 5**: Comparison of FISE_IT1 vs FISE_IT1_GG
6. **Experiment 6**: Impact of mean shift correction

To run a specific experiment, uncomment the relevant section in the file.

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure FastText models are in the `models/` directory
2. **Memory errors**: Reduce `num_traits` or run fewer models simultaneously
3. **Missing datasets**: Verify FISE test files exist in `datasets/{language}/FISE/`

### File Structure Requirements
```
├── models/                          # FastText model files
├── datasets/{language}/FISE/        # FISE test datasets
├── results/{language}/FISE/         # Results output directory
├── fise_experiments.py              # Main experiment runner
├── fise_utils.py                    # FISE implementation
└── requirements.txt                 # Dependencies
```

## Language Support

Currently supported languages:
- **English (en)**: Full FISE_1 dataset
- **Italian (it)**: Multiple test variants with grammatical gender support

## Citation

If you use FISE in your research, please cite the original methodology and this implementation appropriately.