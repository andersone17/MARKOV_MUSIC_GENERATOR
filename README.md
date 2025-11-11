# MARKOV_MUSIC_GENERATOR: Understanding Uncertainty Project 1
## Eddie, Heywood, Finn, Chase, and Harrison

## Research Question and Motivation

### Can we create new music computationally that sounds similar to existing music? 

### This question is important because it will open up a world of possibility for electronic music generation. Traditional artists who physically record music might be threatened at first by this proposition; fearing that they will be replaced by artifical music generators in the near future. To address this very valid concern, we offer the following statement: Our potential music generator will allow current physical artists to utilize computational pattern recognition to spark their inspiratition and give them a base to build off of for future music, which will give them time back that they can spend on the awesome solos that computers cannot generate. We are only trying to generate base chorus chord strings. 

## What are we predicting?

### We are predicting the probability of the next chord given the previous chord(s). This prediction comes from analyzing current music patterns (chord sequences). This prediction allows us to string chords together to create new music. 

## How are we doing this?

### We are doing this using Markov Transiton Models. The defining feature of a Markov Chain, known as the Markov Property, states that the probability of the next state is only dependent on a limited number of previous states, not the entire state history. Thus, we can create a matrix of next-chord sequences based on what chord you are coming from.

## Data Source

### Our data is collected from the Chordonomicon dataset, which is a collection of over 666,000 chord progressions from over 500,000 songs across genres and artists. The creators of the Chordonomicon dataset noticed a lack in large-scale datasets suitable for deep learning applications and limited research exploring chord progressions. They solved thius problem by scraping user-generated progressions and associated metadata to provide valuable insights to the research and machine learning community. Every observation in the Chordonomicon dataset has a chord progression associated with it, meaning when filtering by artist or genre we will never see a song_id without a chord progression. [link](https://arxiv.org/abs/2410.22046)

## Meta data

## Modeling Frame work (Input to Data)


# Recommended Project Steps from Oai
---

## Phase 1 EDA
a. Create a meta data - feature names and label  
b. Analysis features - Descriptive stats, multi var. analysis, Visualization, pattern/trends  

---

## Phase 2 Data Preparation
a. Cleaning: Handle missing values(imputes), duplicates, inconsistencies,  
b. Feature engineering: Domain-specific features, interactions, aggregations  
c. Encoding: One-hot, label encoding, embeddings  
d. Scaling/normalization: Min-Max, StandardScaler, RobustScaler  
e. Features selection - Which variable to use etc. this is extremely important  
Features selection can be PCA, K-nn mean, Lasso, Multi-correlation etc.  

---

## Phase 3 Model
Split - Train/val/test. Train data set is for traing the model, val to validate and ensure no overfiting etc. Test or Out-of-Time data is the final results.  
Modeling  

---

## Phase 4 compare results - ad hoc analysis etc.

---

## Phase 5 Write up

---

## Final
Edit and last minutes change