#!/usr/bin/python3

# Author: Josue Caraballo
#
# Description: This script accepts an input data-set, the split percentages, and creates
#              new test/train data-set files with an similar distributions of labels as
#              the other datasets. Also does some minor preprocessing.
# Assumptions:
#              - The .csv input file text column has the header `text`
#              - The .csv input file label column has the header `label`
#              - In the directory where the .csv file is located, there
#                do NOT exist files <filename>_{train,test}.csv
#
# Command: python3 split.py data.csv <% for train> <% for test>
#
# Parameters:
#              data.csv - Dataset with input text and labels
#              <% for train> - Percentage of input dataset to be used for training data-
#                              -set, must sum to 100 with other percentage, must be int
#              <% for test> - Percentage of input dataset to be used for testing datase-
#                             -t, must sum to 100 with other percentage, must be int

import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Preamble
logging.basicConfig(level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    filename="logs/split.log",
    filemode="w")
errors = {
    "_INVALID_ARGS_":"INVALID NO. OF ARGUMENTS PASSED",
    "_INVALID_FILE_":"INPUT FILE IS INVALID",
    "_INVALID_PERCENTAGES_":"TARGET SPLIT PERCENTAGES ARE INVALID"
}
# End of Preamble

# Auxillary functions
def validate(some_arg, arg="percentage"):
    '''
    Description: Function to validate arguments
    Parameters : some_arg - string - An argument
                 arg - string - Enumerated str,
                 + percentage - test/train perc-
                                -entages
                 + file_path - Path to input .csv
    Returns: Boolean - Whether or not the arg is
                       valid
    '''
    if arg == "percentage":
        # Check if 2 percentages
        if len(some_arg) == 2:

            # Check if both are integers, and if they sum to 100
            running_total = 0
            for k,v in some_arg.items():
                try:
                    running_total += int(v)
                except ValueError as e:
                    logging.warning(e)
                    break
                if running_total == 100:
                    return True
    elif arg == "file_path":
        # Check if file exists
        if os.path.isfile(some_arg):

            # Check if file is a .csv
            if some_arg.lower().endswith(".csv"):
                return True

    return False
# End of Auxillary functions

if __name__ == "__main__":
    # Process Arguments
    # Step 1: Check if 3 Arguments are present

    args = [arg for arg in sys.argv
            if 'python' not in arg and
            '.py' not in arg]
    if len(args) != 3:
        logging.warning(errors["_INVALID_ARGS_"])
        exit()
    percentage = dict()
    file_path = dict()
    file_path["raw"], percentage["train"], percentage["test"] = args

    # Step 2: Check if 3 Arguments are valid
    if not validate(file_path["raw"], arg="file_path"):
        logging.warning(errors["_INVALID_FILE_"])
        exit()
    if not validate(percentage):
        logging.warning(errors["_INVALID_PERCENTAGES_"])
        exit()
    else:
        percentage = {k:int(v) for k,v in percentage.items()}


    # Step 3: Load the input data file, drop any other rows, and remove null rows
    #         also encode the labels
    df = dict()
    cols = dict()
    le = LabelEncoder()
    df["raw"] = pd.read_csv(file_path["raw"])
    cols["all"] = set(df["raw"].columns)
    cols["keep"] = set(["text","label"])
    cols["drop"] = cols["all"] - cols["keep"]
    df["raw"] = df["raw"].drop(cols["drop"], axis=1)
    df["raw"] = df["raw"].dropna().reset_index(drop=True)
    series = pd.Series(le.fit_transform(df["raw"].label), name="label_encoded")
    df["raw"] = pd.concat([df["raw"], series], axis=1)

    # Step 4: Shuffle, and select test/train indices
    df["raw"] = df["raw"].sample(frac=1).reset_index(drop=True) # Shuffle
    x_train, x_test, y_train, y_test = train_test_split(
        df["raw"].text,
        df["raw"].label_encoded,
        train_size = float(percentage["train"])/100,
        test_size = float(percentage["test"])/100,
        stratify = df["raw"].label_encoded
    )
    x_train, x_test = (x_train.reset_index(drop=True),
                       x_test.reset_index(drop=True))
    y_train, y_test = (le.inverse_transform(y_train),
                       le.inverse_transform(y_test))
    y_train, y_test = (pd.Series(y_train, name="label"),
                       pd.Series(y_test, name="label"))
    df["train"] = pd.concat([x_train, y_train], axis=1)
    df["test"] = pd.concat([x_test, y_test], axis=1)

    # Step 5: Store resulting .csv files
    span = file_path["raw"].index(".csv")
    file_path["test"] = "{}_test{}".format(file_path["raw"][:span],
                                           file_path["raw"][span:])
    file_path["train"] = "{}_train{}".format(file_path["raw"][:span],
                                           file_path["raw"][span:])
    # Warning: the step below will overwrite any previously made files
    df["train"].to_csv(file_path["train"])
    df["test"].to_csv(file_path["test"])

