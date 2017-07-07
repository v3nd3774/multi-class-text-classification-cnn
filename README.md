### Project: Classify Claims into PolitiTax Classes
### Highlights:

 - This is a **multi-class text classification (sentence classification)** problem.
 - The purpose of this project is to **classify textual claims into 1/3 PolitiTax Classes**
 - The model was built with **Convolutional Neural Network (CNN)** and **Word Embeddings** on **Tensorflow**.

### Data: [PolitiFact Claims](data/politifact.csv)

 - Input: **text**

    - Example: "In 2006, Planned Parenthood performed more prevention services and cancer screenings than abortions, but in 2013, there were more abortions."
    
 - Output: **label**

     - Example: quantity

### Split Input Data:
 - Description: Splits the input data into test and train sets for subsequent steps.
 - Command: python3 split.py data.csv <% for train> <% for test> 
 - Example: `python3 split.py data/politifact.csv 90 10`

### Train:
 - Command: python3 train.py train_data.csv parameters.json
 - Example: `python3 train.py data/politifact_train.csv parameters.json`
 
 A directory will be created during training, and the best model will be saved in this directory. 

### Predict:
 Provide the model directory (created when running `train.py`) and new data to `predict.py`.
 - Command: python3 predict.py trained_model_directory/ test_data.csv
 - Example: `python3 predict.py trained_model_1479757124/ data/politifact_test.csv`

### References:
 - [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
 - [Multiclass Text Classification CNN](https://github.com/jiegzhan/multi-class-text-classification-cnn)
