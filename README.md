# CineStream Movie Recommendation System

<img width="1536" height="1024" alt="3" src="https://github.com/user-attachments/assets/4171c539-5643-4ce8-ab76-9c006c39eb29" />

## Overview
This project details the development of a sophisticated, personalized Movie Recommendation System for the fictional streaming platform CineStream. The central business challenge identified is choice overload, where an overwhelming content catalog leads to user session abandonment, negatively impacting watch time, customer loyalty, and platform growth. To mitigate this, the project's core objective was to design a machine learning solution that generates the Top 5 most relevant movie recommendations for every user. The project rigorously follows the CRISP-DM framework, utilizing the MovieLens Small Dataset (100k ratings) for training and validation. The technical architecture is a Hybrid Recommender System. The primary component is Collaborative Filtering using Matrix Factorization (SVD), which efficiently learns latent feature patterns from historical ratings to predict user enjoyment. This is strategically augmented with Content-Based Filtering (using TF-IDF on movie metadata) to improve recommendation explainability and mitigate the critical cold-start problem for new users and content. Model performance is evaluated using a dual set of KPIs: Model-Level metrics like RMSE ensure prediction accuracy, while Product-Level metrics, including Precision@5, Recall@5, and NDCG@5, directly quantify the system's business effectiveness. The successful completion of this project demonstrates command of the full data science lifecycle and delivers a production-ready recommendation engine essential for boosting engagement and retention on the CineStream platform.


##  Data Science Team (Group 4)
- Member 1:Diana Aloo Scrum Master  
- Member 2: Catherine Kaino  
- Member 3: June Masolo  
- Member 4: Joram Mugesa  
- Member 5: Edinah Ogoti  



## 1. Business Problem

CineStream faces a critical challenge common to all major streaming services: a large, ever-growing content library creates choice overload, resulting in a high degree of decision paralysis for its users. This friction in content discovery directly translates to detrimental business outcomes: users often spend too much time browsing, fail to find immediately interesting content, and consequently abandon their viewing sessions, leading to reduced watch time and increased customer churn. The business imperative is therefore to replace the generic and ineffective global popularity lists with a precise, personalized curation system.

To address this, the project requires the development of a Top-5 Personalized Movie Recommendation System. This system must be capable of accurately predicting individual user ratings for unseen movies by learning from historical user behavior.

### Business Objectives
1. Predict user ratings for unseen movies.  
2. Generate Top-5 movie recommendations for each user.  
3. Improve engagement by surfacing relevant movies.  
4. Provide explainable recommendations.  
5. Address cold-start issues for new users or new movies.

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/8899c499-4a16-4c72-927c-0de4498f6737" />
<img width="769" height="391" alt="image" src="https://github.com/user-attachments/assets/b57c2420-cad9-4d8a-8a05-544cc1eda5a7" />
<img width="1536" height="1024" alt="3" src="https://github.com/user-attachments/assets/4171c539-5643-4ce8-ab76-9c006c39eb29" />

## 2. Data Understanding

### Overview of the Dataset
For this project, we use the MovieLens 100k dataset, from GroupLens Research, University of Minnesota  
URL: https://grouplens.org/datasets/movielens/latest/, a widely used benchmark dataset for recommendation systems.  
It contains explicit movie ratings from real users and provides enough complexity to build collaborative filtering models.

This dataset includes the following key files:

- ratings.csv – User ratings of movies  
- movies.csv – Movie titles & genres  
- links.csv – External movie database IDs (IMDB, TMDB)  
- tags.csv – User-generated tags (optional)

We primarily use:
- ratings.csv for collaborative filtering  
- movies.csv for content-based or hybrid enhancements



- The datasets have no missing values

- The movie dataset contains 3 columns and 9742 rows

- The rating dataset contains 4 columns and 100836 rows

- The two datasets have common movieId column which will be used to merge the two datasets

- The datasets have no duplicates

### EDA
### Univariate 

We explored each variable individually to understand its distribution and key characteristics. 

#### Ratings Distribution
We want to see how users are rating movies. Are most ratings high, low, or evenly distributed? This helps us understand user behavior and potential bias in the data.

![Movie Rating Distribution](images/output_13_0.png)
    


##### The Key Finding: Positive Skew
The most important observation from this histogram will be its shape: it is typically negatively skewed (or skewed to the left), which means the bulk of the data (the mode) is concentrated on the higher end of the rating scale.

Mode: The tallest bars will  be at 4.0 and 5.0 stars.

Interpretation: This confirms a positive bias in user behavior. Users are more likely to rate movies they enjoyed or are simply more motivated to rate movies they finished and liked. They tend to give good ratings, or they simply avoid rating movies they dislike.

### Bivariate EDA
In this step, we explore relationships between two variables to uncover patterns in the data.

#### Average Rating per Movie
We calculate the average rating for each movie to see if some movies are consistently rated higher than others. This can help the recommendation system understand which movies are most appreciated by users.


    
![png](output_24_0.png)
    


##### The Key Finding: Central Tendency and Consensus
Unlike the raw rating count (which heavily skews toward 4.0 and 5.0), the distribution of average ratings per movie exhibits a shape that is slightly more symmetrical or bell-shaped, but still leans toward the positive side (skewed slightly left/negatively skewed).

Mode/Peak of the distribution falls between 3.0 and 4.0 stars. This represents the "average" consensus rating for the majority of the movies in the catalog.

Interpretation: The cluster in the middle shows that most movies fall into a "moderately good" category in terms of overall quality consensus. This means the model needs to be precise when predicting user preferences within this tightly packed range.

##### **Implications for the CineStream Recommender**
This distribution is crucial for understanding the global bias of the content and for setting up baselines for the SVD Matrix Factorization model.
Sharp Peak (Mode): The model must excel at predicting ratings in the 3.0 to 4.0 range, as this is where most movies reside. Accurate prediction here is vital for low RMSE.
Small Tails (Extremes):	The short tails near 1.0 and 5.0 represent movies that have strong, uniform consensus (either universally disliked or universally loved). These movies provide excellent, high-confidence signals for the SVD model to learn preference extremes.
Overall Mean: The overall mean of this distribution serves as a strong baseline for your recommender. Any effective personalized model must significantly outperform simply predicting this global average rating for every user.


### Multivariate EDA

In this step, we explore relationships involving multiple variables simultaneously. This helps us detect patterns, correlations, or trends that might not be obvious from univariate or bivariate analysis.

####  Correlation of Ratings with User Activity and Movie Popularity
We examine how the number of ratings per movie and per user relate to average ratings. This can reveal whether popular movies or highly active users bias the ratings.



    
![png](output_35_0.png)
    


#### The Key Finding: Weak Positive Correlation
Examining the cell representing the correlation between num_ratings and avg_rating on the heatmap, you will  observe a weak positive correlation coefficient (e.g., in the range of +0.1 to +0.3).
Interpretation: This is a crucial finding. It means that while movies with more ratings (popular movies) do tend to have slightly higher average quality scores, the relationship is not strong. Popularity alone is a poor predictor of true quality consensus.

#### **Implications for the CineStream Recommender**
This result is fundamental to the project's Business Understanding and Modeling strategy, as it justifies the need for a sophisticated personalization model:
Weak Positive	Justification for SVD: Proves that a simple Popularity Baseline (recommending the most-rated movie) is insufficient, as it fails to capture quality that is independent of mass appeal. The model must go beyond popularity.
Disentangling Factors:	The SVD model must learn latent factors that capture true quality and personal taste, separate from the movie's overall rating count.

## 3. Data Preparation

In this section, we prepare the dataset for exploratory data analysis (EDA) and modeling. Data preparation is a crucial step because the quality of the input data directly affects the performance, accuracy, and interpretability of the final recommendation system. 

We begin with basic data cleaning, focusing on auditing missing values, checking data types, and identifying potential data quality issues. This is important because recommendation systems depend heavily on reliable user–item interaction data. After the cleaning stage, we will proceed to feature engineering and various levels of EDA (univariate, bivariate, multivariate).


### User- Movie Interaction Matrix
In this step, we prepare the data for collaborative filtering, a common type of recommendation system.  

Collaborative filtering predicts what a user might like based on ratings from similar users. To do this, we need a **User-Movie Interaction Matrix**:

- **Rows** represent individual users.
- **Columns** represent movies.
- **Values** are the ratings given by users to movies.
- Missing ratings (movies not yet rated by a user) are filled with 0.  

This matrix will allow the recommendation system to compute similarities between users or items, and predict ratings for unseen movies.


## 4.Modeling

In this step, we move to the **Modeling phase** of CRISP-DM. 
We use the prepared data (user-movie matrix and movie content features) to generate movie recommendations.

We will implement a **hybrid recommendation system** in two stages:

1. Content-Based Recommendations
   - Uses movie attributes (title + genres) to find similar movies.
   - Does not rely on user ratings directly.
   
2. Collaborative Filtering (User-Based)
   - Uses the User-Movie interaction matrix.
   - Computes similarity between users based on their ratings.
   - Recommends movies liked by similar users.

Finally, the hybrid system will combine both approaches to provide personalized Top-N recommendations.


### Content-Based Recommendations
In this step, we use the data prepared in the previous steps to generate recommendations.  

We focus on **content-based recommendations** first:  
- For a given movie, we use the **cosine similarity matrix** to find movies with the most similar content (title + genres).  
- We rank these similar movies and recommend the top N to the user.  

This approach does **not** rely on user ratings directly, but purely on item attributes.  
Later, collaborative filtering can be added to combine user preferences for better recommendations.

Content-based filtering recommends items similar to what the user has already liked, based on item attributes.  

Here, we use **movie titles and genres** as features:  
- We combine the movie title and genres into a single text column (`content`).  
- We apply **TF-IDF Vectorization**, which converts text into numerical features by weighting words based on their importance in the dataset.  
- Movies with similar content will have similar TF-IDF vectors, which allows us to compute **cosine similarity** and recommend similar movies.



### Collaborative Filtering (User-Based)

In this step, we enhance the recommendation system by incorporating user-based collaborative filtering. Unlike content-based recommendations, this approach leverages user ratings to find similarities between users and recommend movies that similar users liked. 

Key Concepts:

- User-Movie Interaction Matrix:  
  Rows represent users, columns represent movies, and values are the ratings given by users. Missing ratings are filled with 0.

- User Similarity:  
  We calculate the similarity between users using cosine similarity on their rating vectors. Users with similar tastes are likely to enjoy similar movies.

- Personalized Recommendations:  
  Movies highly rated by similar users but not yet watched by the target user are recommended.

This approach captures personal preferences and complements the content-based system, forming a hybrid recommender.

Why Collaborative Filtering?

•  It does not require detailed movie descriptions

•  It learns patterns directly from user behavior

•  It performs well on structured rating datasets like MovieLens


#### Creating a function for generating a movie recommendation.

In this step, we generate movie recommendations for a specific user based on the ratings of similar users. 

**Key Concepts:**
- Users with similar tastes are identified using the cosine similarity computed in the step above.
- Movies highly rated by similar users but not yet watched by the target user are prioritized.
- This helps personalize recommendations based on user behavior rather than just movie content.


#### Model performance produced 
    RMSE: 0.8786
    MAE:  0.6760


1. Root Mean Square Error (RMSE: 0.8760): On a 5-star rating scale, this means that, on average, the SVD model's prediction is off by approximately 0.876 stars. For example, if a user gave a movie a rating of 4.0, the model might typically predict a rating of around 3.124 or 4.876.
Significance for CineStream: This is a good general accuracy score for Collaborative Filtering models on the MovieLens dataset. It suggests the model is making reasonably accurate predictions, though hyperparameter tuning (finding the optimal number of latent factors and regularization) might still reduce this value further.
2. Mean Absolute Error (MAE: 0.6731): On average, the absolute difference between the true rating and the predicted rating is approximately 0.673 stars.
Significance for CineStream: This score is always lower than the RMSE. Since the difference between RMSE (0.8760) and MAE (0.6731) is relatively small, it indicates that the model does not have many severe, large prediction errors (outliers). If the RMSE were much larger than the MAE, it would suggest a higher number of significant prediction failures.

#### Hyperparameter Tuning
To ensure you find the optimal SVD settings, Grid Search is used to test different combinations of hyperparameters.

After hyperparameter tuning and finding the best parameters the SVD model performed as below;
    RMSE: 0.9026
    MAE:  0.6995

### Hybrid Recommendation System

In this step, we combine **content-based** and **user-based collaborative filtering** to create a hybrid recommendation system.  

**Goal:** Generate personalized Top-N movie recommendations that consider both the similarity of movies (content) and the preferences of similar users (collaborative filtering).

**Approach:**

1. **Content-based filtering**: Uses the TF-IDF cosine similarity of movie titles + genres to find movies similar to those the user has already watched.  
2. **User-based collaborative filtering**: Finds users similar to the target user based on ratings, then recommends movies highly rated by these similar users.  
3. **Hybrid combination**: Merge and rank movies from both approaches to create a final Top-N recommendation list.



    
![png](output_75_1.png)
    



### Hybrid Recommendation System – Stacked Bar Chart Explanation

The stacked bar chart breaks down each movie's recommendation score into two components:

**1. Content-Based Score:**  
- Measures similarity to movies the user has watched.  
- Uses TF-IDF vectors of titles and genres.  
- Captures potential interest based on movie attributes.

**2. Collaborative Filtering Score:**  
- Represents influence from ratings of similar users.  
- Captures personalized preferences based on user behavior.

**Purpose:**  
- Shows how much each movie’s recommendation comes from content vs similar users.  
- Helps interpret whether recommendations rely on content similarity or collaborative influence.  
- Provides transparency and allows quick comparison across Top-N movies.

**Visual Enhancements:**  
- Bars are color-coded into content vs collaborative contributions.  
- Movies are sorted by total hybrid score, highest on top.

**Interpretation:**  
- Larger content segments → similarity-driven recommendations.  
- Larger collaborative segments → user-driven personalization.  
- Balanced segments → effective hybrid recommendation.


## 5.Evaluating the Hybrid Recommendation System

In this step, we aim to quantitatively assess the performance of our hybrid recommendation system. While we have visualizations and top-N recommendations, it is important to measure **how well the recommendations match actual user preferences**.

**Goals:**
1. Evaluate how accurate the hybrid system is in recommending movies that users actually like.
2. Compare hybrid recommendations with content-based only and collaborative filtering only approaches.

**Metrics Used:**
- **Precision@K:** Measures the proportion of recommended movies that are actually relevant to the user.
- **Recall@K:** Measures the proportion of relevant movies that are successfully recommended.
- **F1-Score@K:** Harmonic mean of Precision@K and Recall@K for balanced evaluation.

**Procedure:**
1. For each user, generate Top-N recommendations from the hybrid system.
2. Compare these recommended movies with the movies the user has actually rated highly in the test set.
3. Compute Precision@K, Recall@K, and F1@K for each user and average across all users.

This evaluation will allow us to determine if the hybrid approach improves recommendations compared to pure content-based or collaborative filtering models.


    {'Precision@K': 0.7297208538587848, 'Recall@K': 0.0914745990785698}


**Precision@K 0.7314** (High) Accuracy of Recommendations: This means that when the system gives a user a list of recommendations, approximately 73.14% of those movies are ones the user is highly likely to enjoy. This score is excellent, indicating the system is highly effective at avoiding bad suggestions and delivering relevant content.

**Recall@K 0.0917** (Low)Coverage of Preferences: This means the system only manages to capture about 9.17% of all the movies the user would actually like (in the entire catalog) and place them in the top K list.

### Retrain SVD with Optimal Hyperparameters

After performing hyperparameter tuning using GridSearchCV, we retrain the SVD model with the **best parameters** on the full dataset. This ensures that the final model has learned from all available data, maximizing prediction accuracy.

Once retrained, the optimized SVD model can be used in the hybrid recommendation system to update predicted ratings and improve the personalized recommendations.



### Updating Hybrid Recommendations

With the optimized SVD model, we update the hybrid recommendation system to generate Top-N recommendations for each user. The hybrid scores now incorporate more accurate predicted ratings from SVD, improving personalization.

We will also include content-based scores and collaborative filtering scores to maintain transparency and interpretability of recommendations.


### Visualizing Hybrid Recommendations

In this step, we visually analyze the Top-N recommendations generated by the optimized hybrid system. 

**Purpose of Visualization:**
1. Understand the contribution of each component (Collaborative Filtering vs Content-Based) to the final hybrid score.
2. Identify which movies are recommended mainly due to SVD predictions (personalized user preferences) and which due to content similarity.
3. Provide an interpretable overview of recommendations for each user.

**Approach:**
- For each Top-N recommended movie, calculate:
    - CF_Score: Contribution from SVD predictions.
    - Content_Score: Contribution from cosine similarity of movies the user has rated.
- Combine them in a stacked bar chart.
- Highest total hybrid score appears at the top for clear comparison.


                                   title  \
    0                  D.A.R.Y.L. (1985)   
    1         G.I. Joe: The Movie (1987)   
    2  Batman/Superman Movie, The (1998)   
    3             Children of Men (2006)   
    4                        Home (2015)   
    
                                                  genres  CF_Score  Content_Score  \
    0                          Adventure|Children|Sci-Fi  4.730996      24.645663   
    1  Action|Adventure|Animation|Children|Fantasy|Sc...  4.110667      21.135129   
    2  Action|Adventure|Animation|Children|Fantasy|Sc...  4.138993      21.036888   
    3             Action|Adventure|Drama|Sci-Fi|Thriller  3.999966      21.025195   
    4  Adventure|Animation|Children|Comedy|Fantasy|Sc...  4.231374      20.201773   
    
       Hybrid_Score  
    0     14.688330  
    1     12.622898  
    2     12.587941  
    3     12.512581  
    4     12.216574  

#### Combined Visualization – Top-5 Movie Recommendation Breakdown

This visualization allows quick comparison across the Top-5 movies and shows which component drives each recommendation.

![png](output_101_0.png)
    
### Interpretation:

Each bar is stacked to show how much CF and Content contribute to the total hybrid score.

1.Black dots and lines indicate the final hybrid score for each movie.

2.Taller bars and higher dots mean stronger recommendations.

3.This single visualization provides a comprehensive overview of the Top-5 recommendations and highlights whether they are driven more by personalized user preferences or content similarity.

## 6. Deployment

**Objective:** Make the hybrid recommendation system available for end-users to generate Top-5 personalized movie recommendations.

**Deployment Overview:**

1. **Deployment Options:**
   - **Web Application:** Integrate the model into a web interface where users can input their ratings or select movies they like, and receive Top-5 recommendations.
   - **API Service:** Wrap the recommendation system into a REST API that can be accessed programmatically by other applications.
   - **Batch Processing:** Generate recommendations offline for all users and store results in a database for fast retrieval.

2. **Key Deployment Considerations:**
   - **Input Handling:** Ensure new users or unseen movies are handled gracefully (e.g., cold-start problem).
   - **Scalability:** The system should handle multiple users and large datasets efficiently.
   - **Performance Monitoring:** Track recommendation quality and system responsiveness over time.
   - **Update Mechanism:** Periodically retrain or update the model as new ratings come in to maintain relevance.

3. **User Interaction:**
   - Users provide ratings or interact with a movie catalog.
   - The system calculates hybrid scores using content similarity and SVD predictions.
   - Top-5 personalized recommendations are displayed with movie titles, genres, and optional score breakdowns.

**Outcome:**
- A fully functional recommendation engine delivering personalized movie suggestions, aligning with the project objective of recommending the Top-5 movies a user is likely to enjoy based on their past ratings.


## Conclusion

Overall Conclusion: 
The project demonstrates that data-driven recommendation systems can significantly enhance user experience and business outcomes in digital platforms. By leveraging user ratings, movie metadata, and hybrid modeling techniques, personalized recommendations can be delivered effectively, promoting both user satisfaction and engagement. This project lays a strong foundation for future improvements, including real-time personalization, scalability, and integration with broader digital ecosystems.

Future Scope:
* Expanding to larger datasets for more comprehensive recommendations.  
* Incorporating additional user interaction data (e.g., watch time, search queries).  
* Implementing advanced machine learning models, such as matrix factorization and deep learning-based recommendation systems, to further improve prediction accuracy.

