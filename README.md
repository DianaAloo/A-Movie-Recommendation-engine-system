# A Movie Recommendation Model 
## A data-driven recommendation model built using the MovieLens dataset.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e1a3b9d5-af22-40eb-b43a-0d37b1adab50" />


## Project Overview

This project demonstrates the creation of a **movie recommender system** using Python and the **Surprise library**. 

The main objective was to build a model that provides the **top 5 movie recommendations** for a user, leveraging user-based collaborative filtering and content based filtering techniques. 

The work follows the CRISP-DM framework, covering business understanding, data preparation, modeling, evaluation, and recommendations for deployment.

## Project Team (Group 4)
Member 1: Diana Aloo-Scrum Master
Member 2: Catherine Kaino
Member 3: June Masolo
Member 4: Joram Mugesa
Member 5: Edinah Ogoti

## Business Problem
Users often leave the platform when they cannot quickly identify interesting movies.
The business requires a Top-5 Personalized Movie Recommendation System that predicts what each user is likely to enjoy.

The model should:

Learn from historical ratings
Predict ratings for unseen movies
Recommend the Top-5 most relevant options
### Dataset 
MovieLens Latest Small Dataset (ml-latest-small)
Source: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

## Objectives
- Predict user ratings for unseen movies.
- Generate Top-5 movie recommendations for each user.
- Improve engagement by surfacing relevant movies.
- Provide explainable recommendations.
- Address cold-start issues for new users or new movies.


## Approach
The project followed the Data Science CRISP-DM ,Cross-Industry Standard Process for Data Mining,providing a structured, iterative approach to ensure data projects are well-organized, reproducible, and aligned with business objectives.


For this project we achieved the following:  
=======
# Movie Recommendation Model System


<img width="1536" height="1024" alt="3" src="https://github.com/user-attachments/assets/4171c539-5643-4ce8-ab76-9c006c39eb29" />


## Project Overview
This project demonstrates the creation of a **movie recommender system** using Python and the **Surprise library**, based on user ratings. The main objective was to build a model that provides the **top 5 movie recommendations** for a user, leveraging collaborative filtering techniques taught in class.  

The lab uses the **MovieLens 1M dataset**, which contains user ratings for a wide range of movies.

<img width="1024" height="1536" alt="2" src="https://github.com/user-attachments/assets/a4557b29-3ab6-4d58-aa63-f7550346dccf" />


## Objectives
In this lab, we achieved the following:  
=======
### Business Understanding
This project simulates a real-world scenario for a Data Science team at a streaming service.
The business needs a system that can intelligently recommend movies to users based on their past behavior in order to:
- Improve user experience
- Increase watch time
- Enhance customer loyalty
- Drive platform growth
Users face choice overload, and personalized recommendations reduce friction and increase engagement.

- Build a movie recommendation system using content-based 
- Built a **collaborative filtering model** using **SVD** and compared it to KNN-based models.  
- Performed **cross-validation** to evaluate model performance using RMSE and MAE metrics. 
- Predicted ratings for specific users and movies.  
- Added new user ratings to the system and generated **personalized top-n recommendations**.  
- Created functions to:
  - Collect user ratings for movies (`movie_rater`)  
  - Generate top-n recommendations (`recommended_movies`)  
  


<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/8899c499-4a16-4c72-927c-0de4498f6737" />

<img width="769" height="391" alt="image" src="https://github.com/user-attachments/assets/b57c2420-cad9-4d8a-8a05-544cc1eda5a7" />


The system generates **Top-N movie recommendations** tailored to any selected movie.

The project follows these key steps:
1. **Data Exploration & Preprocessing** – Understand the data, clean missing values, and prepare it for modeling.
2. **Model Building** – Implement a collaborative filtering approach using **Singular Value Decomposition (SVD)** to predict user ratings for movies.
3. **Recommendation Generation** – For a given user, generate a ranked list of the **top 5 recommended movies** they are most likely to enjoy.
4. **Evaluation** – Measure model performance using metrics like RMSE to ensure accurate predictions.
   
###  Data Loading & Cleaning
- Imported `movies.csv` and `ratings.csv` from MovieLens.
- Parsed and cleaned genre information.
- Merged datasets into a unified movie-ratings table.

###  Content-Based Filtering
- Processed movie genres into a **TF-IDF matrix**.
- Computed **cosine similarity** between all movies.
- Built a function to recommend movies based on metadata similarity.

###  Collaborative Filtering (SVD)
- Loaded rating data using `surprise`’s `Dataset` and `Reader`.
- Trained an **SVD model** to predict movie ratings.
- Generated predicted ratings for all users and movies.

###  Hybrid Recommendation Model
- Combined **content-based scores** and **predicted SVD ratings**.
- Calculated a **Hybrid Score = (Similarity Score + Predicted Rating) / 2**.
- Ranked movies to generate **Top-N recommendations**.

---

##  Key Functions

### **1. `recommend_movies(movie_title, n=10)`**
Returns the top-`n` movies most similar to the selected movie based on metadata.

### **2. Hybrid Recommendation Generator**
Uses collaborative filtering + content similarity to give more accurate recommendations.

---

##  Example Output 

### **Top-5 Movie Recommendations**

1. **Last Action Hero (1993)**
2. **G.I. Joe: The Movie (1987)**
3. **Batman/Superman Movie, The (1998)**
4. **Children of Men (2006)**
5. **Home (2015)**



<img width="856" height="424" alt="image" src="https://github.com/user-attachments/assets/a0298e56-dfc3-463e-bf7a-c5f07ca57a81" />

This notebook demonstrates the end-to-end process of building a recommendation system and can serve as a foundation for further enhancements in personalized movie recommendations.

##  Lessons Learned
- Collaborative filtering effectively identifies user preferences based on historical ratings.
- The SVD model can generalize well to recommend movies even for users with limited rating history.
- How to work with real-world datasets (MovieLens).
- Implementing both **content-based and collaborative filtering**.
- Building a **hybrid model** for improved accuracy.
- Using `surprise` for SVD-based recommendation.
- Applying cosine similarity for movie metadata comparison.


## Future Improvements
-Incorporate additional features like movie genres or user demographics for **hybrid recommendations**.
- Add user-profile based recommendations.
- Deploy as an interactive web app using **Streamlit** or **Flask**.
- Include movie posters and descriptions using an API (OMDb or TMDb).
- Implement a **neural network recommender** (e.g., Autoencoders).

## How to Use
1. Load the notebook.
2. Explore the dataset and model implementation.
3. Input a user ID to see the **top 5 recommended movies** for that user.




=======
## Implementation Highlights
1. **Data Preparation**
   - Loaded ratings and movies data into Pandas DataFrames.
   - Cleaned unnecessary columns and converted the dataset into a Surprise-compatible format.

2. **Model Selection**
   - Tested multiple algorithms: `SVD`, `KNNBasic`, `KNNWithMeans`, and `KNNBaseline`.
   - Evaluated models using cross-validation.
   - Selected **SVD** as the best-performing model with optimal hyperparameters (`n_factors=50`, `reg_all=0.05`).

3. **Making Recommendations**
   - Made predictions for individual users and items.
   - Allowed addition of a new user with custom ratings.
   - Ranked all movies by predicted rating and returned the **top 5 movie titles**.



## Example Output
For a new user with ratings provided, the top 5 recommended movies were:  
1. Shawshank Redemption, The (1994)  
2. Philadelphia Story, The (1940)  
3. Rear Window (1954)  
4. Dr. Strangelove or: How I Learned to Stop Worrying (1964)  
5. Amadeus (1984)  

<img width="769" height="391" alt="image" src="https://github.com/user-attachments/assets/09888c46-1aff-499a-b719-c49bfd1088ac" />


<img width="747" height="387" alt="image" src="https://github.com/user-attachments/assets/1f46f8bf-22e5-427f-bded-81737b249004" />


## Future Improvements
- Include **genre-specific recommendations**.  
- Integrate hybrid recommendation methods (combining collaborative and content-based filtering).  
- Deploy as a web application for interactive recommendations.

## Summary
This lab allowed us to implement a **collaborative filtering recommender system** from end to end. We practiced model selection, performance evaluation, and personalized recommendation generation, gaining hands-on experience in building real-world recommendation systems.

