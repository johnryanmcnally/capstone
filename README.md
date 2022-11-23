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
#### Data Cleaning
The cleaning scripts expect the data in a JSON format according to the Recipe1M+ dataset. [parseIngredients.py](https://github.com/mcnaljr/capstone/blob/main/data_cleaning/parseIngredients.py) extracts and tabulates a clean list of the 999 most frequent ingredients. it uses a list of stop words and ingredient mapping to reduce redundant ingredients, which you may or may not want to modify (perhaps depending on if you believe "tomatoes" and "crushed tomatoes" are unique ingredients. [parseTitle.py](https://github.com/mcnaljr/capstone/blob/main/data_cleaning/parseTitle.py) similarly extracts the most frequent title tokens to initialize the vocabulary for the ingredient prediction model.

#### Recipe Recognition
A few sentences about how to run/train the recipe recognition model

#### Ingredient Prediction
A few sentences about how to run/train the ingredient prediction model

#### Cuisine Identification
A few sentences about how to run/train the ingredient prediciton model

# Streamlit App
A few sentences about how the streamlit app is organized

## How to Help
Describe -at a high level- what the constraints are for each model and the process for submitting them for review

