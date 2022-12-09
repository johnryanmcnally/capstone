<p align="center">
  <img src="./images/rec2_logo.png" alt="Rec2 Logo">
  <h1 align="center">Rec<sup>2</sup>: Recipe Recognition</h1>
</p>

This project provides a pipeline of machine learning models that interpret food recipes:
- Image &rarr; Recipe Title (Recipe Recognition)
- Recipe Title &rarr; Ingredient List (Ingredient Prediction)
- Ingredient List &rarr; Food Categories (Cuisine Identification)

These models were initially trained using [Recipe1M+](http://pic2recipe.csail.mit.edu/) and another [dataset](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images) of Epicurious recipes as part of a capstone project for the Master of Applied Data Science at the University of Michigan.

## How to Use
### Simple Inferencing
If you're just interested in testing the existing models and exploring the relationships between various fods and ingredients, we tried to make that easy for you by setting up a hosted [Streamlit app](https://mcnaljr-capstone-streamlitappmain-e4lj1v.streamlit.app/) where you can easily submit your own images and recipes.

### Using the Code
First, download or fork the repo and your dataset of choice (if you are interested in training).
### Data Access Statement
The Recipe1M+ dataset is available at http://pic2recipe.csail.mit.edu/ to those who agree to MIT's terms and conditions.
The Epicurious dataset is available through [Kaggle](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images), which requires a Kaggle account for download. 
#### Data Cleaning
The cleaning scripts expect the data in a JSON format according to the Recipe1M+ dataset. [parseIngredients.py](https://github.com/mcnaljr/capstone/blob/main/data_cleaning/parseIngredients.py) extracts and tabulates a clean list of the 999 most frequent ingredients. it uses a list of stop words and ingredient mapping to reduce redundant ingredients, which you may or may not want to modify (perhaps depending on if you believe "tomatoes" and "crushed tomatoes" are unique ingredients. [parseTitle.py](https://github.com/mcnaljr/capstone/blob/main/data_cleaning/parseTitle.py) similarly extracts the most frequent title tokens to initialize the vocabulary for the ingredient prediction model. [json_to_csv.py](https://github.com/mcnaljr/capstone/blob/main/data_cleaning/json_to_csv.py) will extract both the ingredients and titles from the chosen JSON and save them to a condensed csv file. For the training subset of the Recipe1M+ dataset this process will take ~10 hours. [ing_map_cleaning.py](https://github.com/mcnaljr/capstone/blob/main/data_cleaning/ing_map_cleaning.py) will further clean the ingredients according to a manually created ingredient mapping. This mapping converts tokens like 'all purpose flour' and 'AP flour' to simply 'flour'.

#### Recipe Recognition
[recognition.py](https://github.com/mcnaljr/capstone/blob/main/model_generation/recipe_recognition/recognition.py) is set up to train on a single image of images where the filenames are hyphenated named of the recipes they depict (like the [Epicurious dataset](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)). The script can either train a new model or load weights from a previous training. After training (or if training is skipped), attention map is generated for a test image. Model architecture can be edited by swapping out the image feature extractor, adjusting the parameters of the Captioner declaration, or changing the layer construction entirely.

#### Ingredient Prediction
[createSparceVecs.py](https://github.com/mcnaljr/capstone/blob/main/model_generation/ingredient_prediction/createSparseVecs.py) uses the ingredient list, title token list, and ingredient map to covert the recipes into sparse 1000-length vectors, which are stored as pickled Dataframes for easy training. [ingredient_train.py](https://github.com/mcnaljr/capstone/blob/main/model_generation/ingredient_prediction/ingredient_train.py) is a simple script that builds, trains, and saves the model. [ingredient_test.py](https://github.com/mcnaljr/capstone/blob/main/model_generation/ingredient_prediction/ingredient_test.py) calculates evaluation metrics against a test set and produces plots illustrating the performance for the top 50 ingredients.

#### Cuisine Identification
The cuisine identification portion of this project uses topic vectors from an Latent Dirichlet Allocation (LDA) and a Word2Vec model. If you simply want to train either model use [foodtype_model_train.py](https://github.com/mcnaljr/capstone/blob/main/model_generation/cuisine_identification/foodtype_model_train.py) or [ethnicity_model_train.py](https://github.com/mcnaljr/capstone/blob/main/model_generation/cuisine_identification/ethnicity_model_train.py). If you wish to explore the models and some supporting figures use [foodtype_model.ipynb](https://github.com/mcnaljr/capstone/blob/main/model_generation/cuisine_identification/foodtype_model.ipynb) for the LDA model and [ethnicity_model.ipynb](https://github.com/mcnaljr/capstone/blob/main/model_generation/cuisine_identification/ethnicity_model.ipynb) for Word2Vec.

Once the models have been trained use [create_keyvectors.py](https://github.com/mcnaljr/capstone/blob/main/model_generation/cuisine_identification/create_keyvectors.py) to generate the key vectors which are used in the app and [create_tsne_df.py](https://github.com/mcnaljr/capstone/blob/main/model_generation/cuisine_identification/create_tsne_df.py) to generate the supporting dfs for the t-SNE plot.

#### Streamlit App
The /streamlitapp folder contains all of the scripts and files required to run the web app. [main.py](https://github.com/mcnaljr/capstone/blob/main/streamlitapp/main.py) is the script that controls the appearance of the site and runs and displays data from the other supporting scripts in the directory. All models and data are saved in the /models subdirectory.

## How to Help
The Rec2 team is excited and proud of where our app is today, but would love to see it go much further! We welcome feedback and contributions from the community. The main constraint for the web app is that the model files cannot exceed 75 MB, and ideally should remain under 50 MB. The models must also adhere to a set ingredient list, like [ingredient_999](https://github.com/mcnaljr/capstone/blob/main/streamlitapp/models/json/ingredients_999.json) in order ot maintain model alignment.

