# A Movie Recommendation Model 
## A data-driven recommendation model built using the MovieLens dataset.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e1a3b9d5-af22-40eb-b43a-0d37b1adab50" />


## Project Overview

This project demonstrates the creation of a **movie recommender system** using Python and the **Surprise library**, based on user ratings. 

The main objective was to build a model that provides the **top 5 movie recommendations** for a user, leveraging collaborative filtering techniques taught in class. 

The work follows the CRISP-DM framework, covering business understanding, data preparation, modeling, evaluation, and recommendations for deployment.

## Objectives

For this project we achieved the following:  

- Processed the MovieLens dataset using **Surprise's Reader and Dataset classes**.  
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
1. **Shawshank Redemption, The (1994)**
2. **Paths of Glory (1957)**
3. **Guess Who's Coming to Dinner (1967)**
4. **Neon Genesis Evangelion: The End of Evangelion (1997)**
5. **Three Billboards Outside Ebbing, Missouri (2017)**


<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/421863f1-9786-4cbb-bdf6-daf8b374dd2a" />











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




