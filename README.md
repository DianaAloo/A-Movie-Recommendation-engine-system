# Movie Recommendation System Lab


<img width="1536" height="1024" alt="3" src="https://github.com/user-attachments/assets/4171c539-5643-4ce8-ab76-9c006c39eb29" />


## Project Overview
This project demonstrates the creation of a **movie recommender system** using Python and the **Surprise library**, based on user ratings. The main objective was to build a model that provides the **top 5 movie recommendations** for a user, leveraging collaborative filtering techniques taught in class.  

The lab uses the **MovieLens 1M dataset**, which contains user ratings for a wide range of movies.

<img width="1024" height="1536" alt="2" src="https://github.com/user-attachments/assets/a4557b29-3ab6-4d58-aa63-f7550346dccf" />


## Objectives
In this lab, we achieved the following:  

- Processed the MovieLens dataset using **Surprise's Reader and Dataset classes**.  
- Built a **collaborative filtering model** using **SVD** and compared it to KNN-based models.  
- Performed **cross-validation** to evaluate model performance using RMSE and MAE metrics.  
- Predicted ratings for specific users and movies.  
- Added new user ratings to the system and generated **personalized top-n recommendations**.  
- Created functions to:
  - Collect user ratings for movies (`movie_rater`)  
  - Generate top-n recommendations (`recommended_movies`)  

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


  <img width="769" height="391" alt="image" src="https://github.com/user-attachments/assets/b18e5f67-c15e-4b64-ab34-6e9f041f3703" />


## Example Output
For a new user with ratings provided, the top 5 recommended movies were:  
1. Shawshank Redemption, The (1994)  
2. Philadelphia Story, The (1940)  
3. Rear Window (1954)  
4. Dr. Strangelove or: How I Learned to Stop Worrying (1964)  
5. Amadeus (1984)  

## Future Improvements
- Include **genre-specific recommendations**.  
- Integrate hybrid recommendation methods (combining collaborative and content-based filtering).  
- Deploy as a web application for interactive recommendations.

## Summary
This lab allowed us to implement a **collaborative filtering recommender system** from end to end. We practiced model selection, performance evaluation, and personalized recommendation generation, gaining hands-on experience in building real-world recommendation systems.
