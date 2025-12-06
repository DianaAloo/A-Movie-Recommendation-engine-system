# CineStream Movie Recommendation System
## A data-driven recommendation model built using the MovieLens dataset.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e1a3b9d5-af22-40eb-b43a-0d37b1adab50" />


## Project Overview
This project details the development of a sophisticated, personalized Movie Recommendation System for the fictional streaming platform CineStream. The central business challenge identified is choice overload, where an overwhelming content catalog leads to user session abandonment, negatively impacting watch time, customer loyalty, and platform growth. To mitigate this, the project's core objective was to design a machine learning solution that generates the Top 5 most relevant movie recommendations for every user. The project rigorously follows the CRISP-DM framework, utilizing the MovieLens Small Dataset (100k ratings) for training and validation. The technical architecture is a Hybrid Recommender System. The primary component is Collaborative Filtering using Matrix Factorization (SVD), which efficiently learns latent feature patterns from historical ratings to predict user enjoyment. This is strategically augmented with Content-Based Filtering (using TF-IDF on movie metadata) to improve recommendation explainability and mitigate the critical cold-start problem for new users and content. Model performance is evaluated using a dual set of KPIs: Model-Level metrics like RMSE ensure prediction accuracy, while Product-Level metrics, including Precision@5, Recall@5, and NDCG@5, directly quantify the system's business effectiveness. The successful completion of this project demonstrates command of the full data science lifecycle and delivers a production-ready recommendation engine essential for boosting engagement and retention on the CineStream platform.

## Project Team (Group 4)
Member 1: Diana Aloo-Scrum Master
Member 2: Catherine Kaino
Member 3: June Masolo
Member 4: Joram Mugesa
Member 5: Edinah Ogoti

<img width="1536" height="1024" alt="3" src="https://github.com/user-attachments/assets/4171c539-5643-4ce8-ab76-9c006c39eb29" />

## Business Problem
CineStream users often abandon their viewing session when they cannot quickly find interesting content, leading to reduced watch time and customer churn. The business requires a system that can intelligently curate content.
###  Goal
To build a machine learning model that generates the Top 5 personalized movie recommendations for any given CineStream user, thereby improving the user experience and increasing engagement metrics like watch time.

The model should:
Learn from historical ratings
Predict ratings for unseen movies
Recommend the Top-5 most relevant options
### Dataset 
MovieLens Latest Small Dataset (ml-latest-small)
Source: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
Content: Contains user ratings, movie titles, and metadata essential for training a collaborative filtering or hybrid recommendation model.

### Objectives
- Predict user ratings for unseen movies.
- Generate Top-5 movie recommendations for each user.
- Improve engagement by surfacing relevant movies.
- Provide explainable recommendations.
- Address cold-start issues for new users or new movies.

### Methodology (CRISP-DM)
The project followed the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

**Business Understanding**: Defined the need for personalized recommendations to solve choice overload and drive CineStream's platform growth.
**Data Understanding**: Explored the MovieLens dataset, analyzing user rating patterns and movie characteristics.
**Data Preparation**: Cleaned, processed, and engineered features from the raw data.
**Modeling**: Developed and tuned a recommendation model (likely Collaborative Filtering and/or Hybrid) to predict user ratings.
**Evaluation**: Used metrics like Root Mean Square Error (RMSE) for prediction accuracy and business metrics (watch time, click-through rates on CineStream) for impact assessment.
**Deployment** (Conceptual): Defined the steps for integrating the final model into CineStream's production environment.


<img width="1536" height="1024" alt="3" src="https://github.com/user-attachments/assets/4171c539-5643-4ce8-ab76-9c006c39eb29" />


<img width="1024" height="1536" alt="2" src="https://github.com/user-attachments/assets/a4557b29-3ab6-4d58-aa63-f7550346dccf" />

## Business Understanding
This project simulates a real-world scenario for a  **Data Science team** at  CineStream streaming service.
CineStream needs a system that can intelligently recommend movies to users based on their past behavior in order to:  
- Improve user experience  
- Increase watch time  
- Enhance customer loyalty  
- Drive platform growth

Users face *choice overload*, and personalized recommendations reduce friction and increase engagement.


<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/8899c499-4a16-4c72-927c-0de4498f6737" />

<img width="769" height="391" alt="image" src="https://github.com/user-attachments/assets/b57c2420-cad9-4d8a-8a05-544cc1eda5a7" />

## Data Preparation

In this section, we prepared the dataset for exploratory data analysis (EDA) and modeling.

The Data Preparation phase was a critical and meticulous step, focused on transforming the raw MovieLens Small Dataset (100k ratings) into the optimized formats required by the proposed Hybrid Recommender System.

First, the core data files—ratings.csv and movies.csv—were ingested and merged on the common movieId field. This step consolidated all explicit user ratings (1-5 stars) with their corresponding movie titles and genre tags. An essential check for missing values and duplicates was performed to ensure data integrity, yielding a clean dataset of user-movie interactions.

For the Collaborative Filtering (SVD) component, the merged data was transformed into a sparse User-Item Interaction Matrix. User IDs and Movie IDs were mapped to create a pivot table where cells contained the rating, forming the foundational input for learning latent preference patterns.

For the Content-Based Filtering component, the metadata required significant feature engineering. The multi-label genres column was split and tokenized, and the combined textual features (title and genres) were processed using TF-IDF Vectorization to create content feature vectors. This vectorization was key to calculating movie-to-movie similarity and mitigating the cold-start problem by providing recommendations even without prior rating history.

Finally, the processed interaction matrix was partitioned into a training set for model fitting and a held-out test set for evaluation, ensuring the model's performance against RMSE and Precision@5 metrics was reliable and unbiased. This rigorous preparation ensured that the data was ready to train a robust hybrid model capable of serving the CineStream platform.

###  Univariate EDA

In this step, we explored each variable individually to understand its distribution and key characteristics. 

#### 1. Ratings Distribution
We want to see how users are rating movies. Are most ratings high, low, or evenly distributed? This helps us understand user behavior and potential bias in the data.

![png](output_12_0.png)

 Key Finding
The most important observation from this histogram is the shape: it is typically negatively skewed (or skewed to the left), which means the bulk of the data (the mode) is concentrated on the higher end of the rating scale.

Mode: The tallest bars are at 4.0 and 5.0 stars.

Interpretation: This confirms a positive bias in user behavior. Users are more likely to rate movies they enjoyed or are simply more motivated to rate movies they finished and liked. They tend to give good ratings, or they simply avoid rating movies they dislike.

    ### Bivariate EDA

In this step, we explored relationships between two variables to uncover patterns in the data.

#### 1. Average Rating per Movie
We calculate the average rating for each movie to see if some movies are consistently rated higher than others. This can help the recommendation system understand which movies are most appreciated by users.


![png](output_14_0.png)

 Key Findings
Unlike the raw rating count, which heavily skews toward 4.0 and 5.0, the distribution of average ratings per movie typically exhibits a shape that is slightly more symmetrical or bell-shaped, but still leans toward the positive side (skewed slightly left/negatively skewed).

Mode/Peak: The distribution's peak  falls between 3.0 and 4.0 stars. This represents the "average" consensus rating for the majority of the movies in the catalog.

Interpretation: The cluster in the middle shows that most movies are rated moderately well. It indicates that the dataset contains a good spread of quality, but overall, the average movie in the catalog is viewed positively or neutrally, rather than negatively.
    
### Multivariate EDA

In this step, we explored relationships involving multiple variables simultaneously. This helps us detect patterns, correlations, or trends that might not be obvious from univariate or bivariate analysis.

#### 1. Correlation of Ratings with User Activity and Movie Popularity
We examine how the number of ratings per movie and per user relate to average ratings. This can reveal whether popular movies or highly active users bias the ratings.

Key Finding: Weak Positive Correlation
The cell in the heatmap representing the correlation between num_ratings and avg_rating, shows a weak positive coefficient (often ranging from approximately +0.10 to +0.30 in the MovieLens dataset).

Interpretation: This weak positive correlation means that while movies with more ratings (popular movies) tend to have slightly higher average ratings, the relationship is not strong. Popularity is not a guarantee of high quality, and many high-quality movies may not be widely rated.


## Preparing Data for the Recommendation System

###  Methods
The primary technical solution utilizes Collaborative Filtering via Matrix Factorization (SVD) to learn latent patterns in user preferences from the 100k explicit ratings in the MovieLens dataset. This was critically augmented by Content-Based Filtering, which used TF-IDF Vectorization on movie titles and genres. This hybrid strategy ensures robust prediction accuracy while addressing limitations such as the cold-start problem and improving explainability. Baselines, including Popularity and Item-based KNN models, were also utilized for performance comparison.

Results and Evaluation
Success for the recommendation system is measured using a dual set of Key Performance Indicators (KPIs). Model-Level KPIs like RMSE and MAE quantify the accuracy of predicted user ratings, ensuring reliable data output. Product-Level KPIs are used to assess the direct business impact, including Precision@5, Recall@5, and NDCG@5, which evaluate the quality and ranking of the final Top-5 recommendation list. Furthermore, metrics like Coverage, Diversity, and Novelty are tracked to guarantee the system promotes varied, interesting, and platform-wide content consumption, ultimately aiming to increase user engagement and drive platform growth for CineStream.

##Modeling 

In this step, we used the prepared data (user-movie matrix and movie content features) to generate movie recommendations.

We  implemented a hybrid recommendation system in two stages:

Content-Based Recommendations

Uses movie attributes (title + genres) to find similar movies.
Does not rely on user ratings directly.
Collaborative Filtering (User-Based)

Uses the User-Movie interaction matrix.
Computes similarity between users based on their ratings.
Recommends movies liked by similar users.
Finally, the hybrid system will combine both approaches to provide personalized Top-N recommendations.

### Grouped Bar Chart: CF vs Content vs Hybrid Scores

**Top-5 Recommendations Score Breakdown**  
This grouped bar chart compares the contribution of Collaborative Filtering (CF), Content-Based, and Hybrid scores for each of the Top-5 recommended movies.  
It provides an intuitive understanding of how each component influences the final recommendation score.


![png](output_82_0.png)
    


### Combined Visualization – Top-5 Movie Recommendation Breakdown

This visualization allows quick comparison across the Top-5 movies and shows which component drives each recommendation.

    
![png](output_88_0.png)
    


# Interpretation:

Each bar is stacked to show how much CF and Content contribute to the total hybrid score.

1.Black dots and lines indicate the final hybrid score for each movie.

2.Taller bars and higher dots mean stronger recommendations.

3.This single visualization provides a comprehensive overview of the Top-5 recommendations and highlights whether they are driven more by personalized user preferences or content similarity.




###  Example Output 

#### **Top-5 Movie Recommendations**

1. **Last Action Hero (1993)**
2. **G.I. Joe: The Movie (1987)**
3. **Batman/Superman Movie, The (1998)**
4. **Children of Men (2006)**
5. **Home (2015)**

<img width="856" height="424" alt="image" src="https://github.com/user-attachments/assets/a0298e56-dfc3-463e-bf7a-c5f07ca57a81" />

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

## How to Use
1. Load the notebook.
2. Explore the dataset and model implementation.
3. Input a user ID to see the **top 5 recommended movies** for that user.

### Example Output
For a new user with ratings provided, the top 5 recommended movies were:  
1. Shawshank Redemption, The (1994)  
2. Philadelphia Story, The (1940)  
3. Rear Window (1954)  
4. Dr. Strangelove or: How I Learned to Stop Worrying (1964)  
5. Amadeus (1984)  

<img width="769" height="391" alt="image" src="https://github.com/user-attachments/assets/09888c46-1aff-499a-b719-c49bfd1088ac" />


<img width="747" height="387" alt="image" src="https://github.com/user-attachments/assets/1f46f8bf-22e5-427f-bded-81737b249004" />

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


## Conclusion 
The project demonstrates that **data-driven recommendation systems can significantly enhance user experience and business outcomes** in digital platforms. By leveraging user ratings, movie metadata, and hybrid modeling techniques, personalized recommendations can be delivered effectively, promoting both user satisfaction and engagement. This project lays a strong foundation for future improvements, including real-time personalization, scalability, and integration with broader digital ecosystems.
#### Business Impact
* **Improved User Experience**: Reduced friction in content selection on CineStream.
*  **Increased Retention**: Personalized suggestions keep users subscribed and engaged longer.
* **Revenue Growth**: Directly supports increased watch time, leading to higher subscription and advertising revenue for CineStream.

