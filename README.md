# Recommender Systems — MovieLens 10M

Matrix Factorization with SGD and Collaborative Filtering on the MovieLens 10M dataset.
---

## 🎬 Dataset

Not included due to size. Download here:

👉 [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

After downloading extract and place in a `ml-10m/` folder next to the notebooks:

---

## 📋 Project Steps

### Q1 — Data Loading & Exploratory Analysis
- Load `ratings.dat` and `movies.dat` into DataFrames
- Merge on `MovieID` and create a `Year` column from Unix Timestamp
- Explore rating distribution (scale: 0.5 – 5.0, mean = 3.51)

### Q2 & Q3 — Genre Rating Decline Analysis
- Find the top 5 genres with the biggest average annual rating decrease
- Apply a minimum threshold of ≥1,000 ratings/year to eliminate statistical bias
- **Finding:** Horror, War, Documentary, Romance and Children show the most decline

### Q4a — Train Dataset
- Ratings submitted **before 2008** → 9,211,829 ratings (years 1995–2007)

### Q4b — Test Dataset
- Ratings submitted **during 2008 and 2009** → ~788,225 ratings

### Q4c — Recommendation Model (Part 1 vs Part 2)

**Part 1 — Standard Approach:**
- Vectorized SGD Matrix Factorization
- Sparse matrix, mini-batch sampling, latent factor updates

**Part 2 — Extended Approach:**
- User & Movie filtering (≥20 ratings per user, ≥50 per movie)
- Train/Validation/Test chronological split (80/20)
- Bias terms: `pred = μ + user_bias + movie_bias + dot product`
- Early stopping with patience=10
- Systematic hyperparameter tuning

### Q4d — RMSE Results

| Model | RMSE | Improvement |
|---|---|---|
| Part 1 Baseline | 1.2197 | — |
| Part 1 Tuned | 1.0242 | 16% ✅ |
| **Part 2 Best** | **0.8104** | **33.6%** 🎉 |

### Q5 — Cold Start Problem
- New user with no history → cannot use collaborative filtering
- Solution: **Bayesian Weighted Rating** recommends most popular highly-rated movies
- Top recommendation: Shawshank Redemption (4.43)

### Q6 — Item-Based Collaborative Filtering
- User likes: Iron Man, 300, Transformers
- Cosine similarity on user-movie matrix finds the 10 most similar movies
- Top recommendation: V for Vendetta (2006)

---

## ⚙️ Requirements

pip install pandas numpy scipy scikit-learn tqdm matplotlib

## 👤 Author

**Panos Bekirakis**  


