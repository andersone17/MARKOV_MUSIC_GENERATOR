# MARKOV_MUSIC_GENERATOR: Understanding Uncertainty Project 1
## Eddie, Heywood, Finn, Chase, and Harrison


## For Terry and Oai:
"project_questions.md" has our written answers.
"markov_music_generator.ipynb" has our project code. 


## Research Question and Motivation
Can we create new music computationally that sounds similar to existing music? 

### Question Significance: 
This question is important because it will open up a world of possibility for electronic music generation. Traditional artists who physically record music might be threatened at first by this proposition; fearing that they will be replaced by artifical music generators in the near future. To address this very valid concern, we offer the following statement: Our potential music generator will allow current physical artists to utilize computational pattern recognition to spark their inspiratition and give them a base to build off of for future music, which will give them time back that they can spend on the awesome solos that computers cannot generate. We are only trying to generate base chorus chord strings.

## Prediction
We are predicting the probability of the next chord given the previous chord(s). This prediction comes from analyzing current music patterns (chord sequences). This prediction allows us to string chords together to create new music. 

## Method
We are predicting the future chord sequence using Markov Transiton Models. The defining feature of a Markov Chain, known as the Markov Property, states that the probability of the next state is only dependent on a limited number of previous states, not the entire state history. Thus, we can create a matrix of next-chord sequences based on what chord you are coming from.

## Data Source
Our data is collected from the Chordonomicon dataset, which is a collection of over 666,000 chord progressions from over 500,000 songs across genres and artists. The creators of the Chordonomicon dataset noticed a lack in large-scale datasets suitable for deep learning applications and limited research exploring chord progressions. They solved thius problem by scraping user-generated progressions and associated metadata to provide valuable insights to the research and machine learning community. Every observation in the Chordonomicon dataset has a chord progression associated with it, meaning when filtering by artist or genre we will never see a song_id without a chord progression. [Link](https://arxiv.org/abs/2410.22046)

## Data Structure
The columns of our data are as follows: id, chords, release date, genres, decade, rock genre, artist id, main genre, spotify song id, and spotify artist id. We will be ignoring most of these columns as we are only concered with id, chords, and genres. This is for our first attempt at producing novel sequences that form full music, which will be based on genre (rock and potentially jazz at first). We are considering transitioning this attempt based on its success, or lack thereof, to artist-specific chord transition pattern recognition to create new music that fits in an artist's normal flow. 

## Data Limitations
Something interesting we noticed is that the authors of the dataset compressed identical chords into one single chord. For example, if the original sequence was A-G-G-A, the sequence the authors stored was simply A-G-A. This could greatly change the way the msuic sounds by removing long similar melodic strings. 