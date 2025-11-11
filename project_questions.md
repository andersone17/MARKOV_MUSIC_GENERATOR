# Project 1: Markov Model for Music Generation
## Eddie, Heywood, Chase, Finn, and Harrison

---

## Goal

Non-parametrically model chord progressions and generate new chord sequences (music). Option we chose:

- Chordonomicon: 680,000 chord progressions of popular music songs. Create a chord generator, similar to what we did with Bach in class, but for a particular artist or genre.  
  [https://github.com/spyroskantarelis/chordonomicon](https://github.com/spyroskantarelis/chordonomicon)

---

### 1. Describe the data clearly -- particularly any missing data that might impact your analysis -- and the provenance of your dataset. Who collected the data and why? (10/100 pts)

Our data is collected from the Chordonomicon dataset, which is a collection of over 666,000 chord progressions from over 500,000 songs across genres and artists. Chord progressions encapsulate important information about music, and are generally all an artist needs in order to learn and play an existing song. However, the creators of the Chordonomicon dataset noticed a lack in large-scale datasets suitable for deep learning applications and limited research exploring chord progressions. Thus, by scraping various sources of user-generated progressions and associated metadata the Chordonomicon dataset was compiled with hopes of providing valuable insights to the research and machine learning community. In terms of missing data, there is not a concern as every observation in the Chordonomicon dataset has a chord progression associated with it, meaning when filtering by artist or genre we will never see a song_id without a chord progression. For further reading on the Chordonomicon dataset and its applications, see this [link](https://arxiv.org/abs/2410.22046).

---

### 2. What phenomenon are you modeling? Provide a brief background on the topic, including definitions and details that are relevant to your analysis. Clearly describe its main features, and support those claims with data where appropriate. (10/100 pts)

In our "Markov Music Generator," the phenomenon we are modeling are chord progressions in the style of any particular Spotify artist or music genre using Markov Chains. A Markov Chain is a probabilistic system that moves between states, where each transition has a certain likelihood. In our case, the probability of the 'next' chord depends on the previous k-chords, where k is the order. This is the defining feature of a Markov Chain, known as the Markov Property, which states that the probability of the next state is only dependent on a limited number of previous states, not the entire state history.

There are different order markov chains. We evaluated orders 1, 2, and 3. In a first-order Markov Chain, the probability of the next chord depends only on the chord immediately before it. Thus, P(next chord) = P(next chord | current chord). Basically, we are trying to find patterns where one chord commonly follows another. In a second-order Markov Chain, the next chord depends on the probability of the previous two chords. P(next chord) = P(next chord | previous chord, chord before that). Same for the third-order chain but with three previous states as given dependencies.

Markov Chains work well for music generation because chord progressions are not random. Certain chords commonly follow others (for example, V to I cadences in tonal harmony). These patterns differ across artists and genres, but within a subset (e.g. our specific rock or jazz genres) there are observable patterns.

---

### 3. Describe your non-parametric model (empirical cumulative distribution functions, kernel density function, local constant least squares regression, Markov transition models). How are you fitting your model to the phenomenon to get realistic properties of the data? What challenges did you have to overcome? (15/100 pts)

**Model Features**  
- **State Space**: Our state space is defined as the unique set of chords for a given artist or genre.  
- **Order**: We experiment with different sequential dependencies to get the most accurate harmonic forecast for a given artist/genre.  
  - **1st-Order Dependence**: First order dependence tends to modulate across many musical keys because many chords are not specific to any given key. For example, an E chord could function as the IV, or V chord of musical keys E, B, and A respectively. We have very little anchor to one key in this case, and the result is a more 'avante-garde' tone.  
  - **2nd-Order Dependence**: Second order dependence generally adds enough context to stabilize to a major key and its relative minor. We see key changing, but not frequently with most pop/rock artists. Jazz is the exception where second order dependence is not enough to stabilize the chord forecast.  
  - **3rd-Order Dependence**: With third-order dependence, we essentially "overfit" our music generator to an artist's songs, and reproduce known chord progressions from those songs, rather than 'new' harmonies and chord progressions. This is a tradeoff, as we stabilize to a tonal quality that is pleasing, but fail to produce something 'new.' The exception is for music genres that modulate between keys more often, such as jazz; these genres require higher sequential dependence such as order 3.  
- **Transition Matrix**: Based on the order of our sequential dependence, we create a transition matrix that contains the probability of transitioning from one state to another.

---

### 4. Either use your model to create new sequences (if the model is more generative) or bootstrap a quantity of interest (if the model is more inferential). (15/100 pts)

*Please see our work in `project1.ipynb` for the code used to generate new chord sequences based on our Markov transition model.*

---

### 5. Critically evaluate your work in part 4. Do your sequences have the properties of the training data, and if not, why not? Are your estimates credible and reliable, or is there substantial uncertainty in your results? (15/100 pts)

*As stated previously, we do employ some of the training data in order to account for patterns that might be missing in the data. For example, Jazz is famous for its improvisation and so improvisation translates to an absence or lack thereof patterns. This deficit can be handled by relying on some of the patterns in the training data and so some of the chords were not generated by the markov transition matrix. Despite this issue with the jazz notes our work is credible and reliable due to the fact that it has a consistent pattern that is not erratic and this can be heard when the chords are played.*

---

### 6. Write a conclusion that explains the limitations of your analysis and potential for future work on this topic. (10/100 pts)

**Limitations and Future Work**: One limitation of our Markov Music Generator is that it does not account for rhythm, melody, or dynamics, which are crucial elements of music composition. Additionally, the model assumes that chord transitions are solely dependent on the previous k-chords, which may oversimplify the complex relationships in music theory. Future work could involve integrating additional musical features, such as rhythm patterns or melodic contours, into the model. Furthermore, exploring higher-order Markov models or incorporating machine learning techniques could enhance the generator's ability to produce more nuanced and musically coherent compositions.

---

### 7. Github Repo

**GitHub Repo**: Our GitHub repository can be found [here](https://github.com/andersone17/MARKOV_MUSIC_GENERATOR).