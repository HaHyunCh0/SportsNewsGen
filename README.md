# SportsNewsGen
# For NLP Project
Generate news article from sports match results

## Data Setup
1. Download dataset from Kaggle https://www.kaggle.com/datasets/irkaal/english-premier-league-results
2. Rename the dataset file, "raw.csv" and move it to the data/ directory
3. Run utils/clean_data.py and obtain "raw_cleaned.csv"
4. Locate the new file into the misc/ directory and run add_reference_to_raw_data.ipynb
5. Convert the file into a tsv file and rename it with "data_with_reference.tsv"
6. Locate the file into the data/ directory
7. Run utils/create_inputs.py to generate the split dataset(train, val, and test)
8. You may modify the DATA_PATH and the comments in the code to create each different trial set


## Data Directories
- Datasets can be found in the data/ directory
- The folders from 1 to 6 are used for the following trials:
1) baseline
2) adding FTR and HTR tokens
3) adding FTR and HTR tokens and the bias token
4) extending abbreviated tokens into fullname
5) adding fulltime and halftime result tokens from 4)
6) adding the result tokens and the bias token from 4)


## Train
1. Enter 'pip install -r requirement.txt' to install dependent packages
2. Run run_train.sh to train and test (can edit the script to change hyperparameters)
3. Evaluation results and predictions will be stored in the result directory declared in the run_train.sh

