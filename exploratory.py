# Project 1
#### Data Collection and Cleaning


# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import music21
from music21 import harmony
import networkx as nx

# Load dataset
df = pd.read_csv("hf://datasets/ailsntua/Chordonomicon/chordonomicon_v2.csv")

def get_songs(data):
  songs = data["chords"]

  #remove any <verse>/<chorus> tags from the chord data
  cleaned_songs = [re.sub(r'<[^>]+>', '', song).strip() for song in songs]
  #collapse multiple spaces into one
  cleaned_songs = [' '.join(song.split()) for song in cleaned_songs]

  #convert from string of chords to list where each index is chord
  songs = [song.split() for song in cleaned_songs]

  return songs


# #### State Space
def get_states(songs, order=1):
  if order != 1:
    higher_order_songs = []
    for song in songs:
      higher_song = ['_'.join(song[(t-order-1):(t-1)]) for t in range(order+1, len(song))]
      higher_order_songs.append(higher_song)
    higher_order_states = set()
    for song in higher_order_songs:
      new_states = set(song)
      higher_order_states = higher_order_states.union(new_states)

    higher_order_states = list(higher_order_states)
    return higher_order_states, higher_order_songs
  
  else:
    states = set()
    for song in songs:
      new_states = set(song)
      states = states.union(new_states)

    states = list(states)
    return states

#### Transition matrix and Heat Map
def transition_matrix_and_heat_map(states, songs):
    S = len(states)

    # Initialize transition count matrix
    tr_counts = np.zeros((S, S))

    # Count transitions across all trajectories
    for song in songs:
        for t in range(1, len(song)):
            prev_state = song[t-1]
            curr_state = song[t]
            i = states.index(prev_state)
            j = states.index(curr_state)
            tr_counts[j, i] += 1

    # Compute total outgoing transitions per state
    sums = tr_counts.sum(axis=0, keepdims=True)

    # Normalize to get transition probabilities (columns sum to 1)
    tr_pr = np.divide(
        tr_counts, 
        sums, 
        out=np.zeros_like(tr_counts), 
        where=sums != 0
    )

    # Convert to DataFrame for readability
    TM = pd.DataFrame(np.round(tr_pr, 3), index=states, columns=states)

    plt.figure(figsize=(13, 11))
    sns.heatmap(tr_pr, 
                cmap='Blues',
                square=True,          
                xticklabels=states,
                yticklabels=states,
                cbar_kws={'label': 'Transition Probability'})

    plt.title('Transition Probabilities')
    plt.xlabel('...From State')
    plt.ylabel('To State...')
    plt.show()

    return (TM, tr_pr)

#### Clean Chords for Music 21
MANUAL_MAPPINGS = {
    'Cminmaj7': 'CmM7',
    'Fsminmaj7': 'F#mM7',
}
def normalize_chord_string(chord_str: str) -> str:
    """
    Cleans up chord names to match what music21 expects.  
    Args:
        chord_str: A string containing a chord name
    Returns:
        A string containing a normalized chord name
    """
    original_str = chord_str.strip()
    
    #Check if this chord has a manual change
    if original_str in MANUAL_MAPPINGS:
        return MANUAL_MAPPINGS[original_str]
    normalized = original_str
    
    #Convert flat notes
    normalized = normalized.replace('Bb', 'B-')
    normalized = normalized.replace('Eb', 'E-')
    normalized = normalized.replace('Ab', 'A-')
    normalized = normalized.replace('Db', 'D-')
    normalized = normalized.replace('Gb', 'G-')
    
    #capitalize 'sus' terms so they don’t get changed in the next step.
    normalized = normalized.replace('sus4', 'SUS4')
    normalized = normalized.replace('sus2', 'SUS2')
    normalized = normalized.replace('sus', 'SUS')
    
    #sharp notation
    normalized = normalized.replace('s', '#')
    
    normalized = normalized.replace('maj#9', 'M#9')
    normalized = normalized.replace('maj7', 'M7')
    normalized = normalized.replace('maj9', 'M9')
    normalized = normalized.replace('min', 'm')
    normalized = normalized.replace('no3d', '5')

    #'7sus2' convert to '7add(sus2)'
    normalized = normalized.replace('7SUS2', '7add(sus2)')
    normalized = normalized.replace('M7SUS2', 'M7add(sus2)')
    
    #lowercase for suspended chords
    normalized = normalized.replace('SUS4', 'sus4')
    normalized = normalized.replace('SUS2', 'sus2')
    normalized = normalized.replace('SUS', 'sus')
    
    return normalized

#### Random Chord Generator
def generate_chords(states, n_sim, tr_pr, initial_state=None):

    states = [normalize_chord_string(s) for s in states]

    if not initial_state:
        ## Random chords:
        np.random.seed(100) 
        initial_state = np.random.choice(states) # Choose an initial state at random
        state_index = states.index(initial_state) # Get the index of the initial state
        print(f'Initial state: {initial_state}')
    else:
        state_index = states.index(initial_state)
        print(f'Initial state: {initial_state}') 

    simulation = [initial_state]
    for t in range(n_sim-1): 
        pr_t = tr_pr[:,state_index] # Transition probabilities at this state
        state_index = np.random.choice(len(states), p=pr_t) # Choose new state index
        simulation.append(states[state_index]) # Append new state to simulation

    new_chorale = [state.split()[-1] for state in simulation] # Convert to chords

    return new_chorale

#### Separate Chords for Higher Order Chains 
def flatten_chorale(new_chorale, num_chords):
    flat = []
    first = True
    
    for element in new_chorale:
        chords = element.split('_', num_chords - 1)  # safe even if num_chords > 3
        
        if first:
            flat.extend(chords)
            first = False
        else:
            flat.append(chords[-1])
    
    return flat

#### Look at connected components to find a good starting chord
def connected_components(tr_pr):
    G = nx.from_numpy_array(tr_pr, create_using=nx.DiGraph()) # Create directed graph in nx
    is_connected = nx.is_strongly_connected(G) # Test connectivity
    print(f"Connected: {is_connected}")

    # Get connected components if not connected
    if not is_connected:
        strong_components = list(nx.strongly_connected_components(G))
        print(f"Number of strongly connected components: {len(strong_components)}")
        return strong_components
    else:
        return []
    
def components_to_states(states, strong_components, idx):
    component_states = [states[i] for i in strong_components[idx]]
    component_states = [normalize_chord_string(s) for s in component_states]
    return component_states

#### Play a given generated chord progression using Music 21
def play_music(new_chorale):
  stream = music21.stream.Stream()
  for chord_symbol in new_chorale:
    chord = music21.harmony.ChordSymbol(chord_symbol)
    chord.duration = music21.duration.Duration(2.0)  
    stream.append(chord)
  stream.write('midi', fp='music.mid')
  stream.show('midi')  




#####################################################################################################
#####################################################################################################
#####################################################################################################
### Rock Genre Markov Chain

# INPUTS
ROCK_GENRE = 'classic rock'
ORDER = 2
INITIAL_STATE = 'Am_C'

df['rock_genre'] = df['rock_genre'].astype(str).str.lower()
df['spotify_artist_id'] = df['spotify_artist_id'].astype(str)
rock = df[df['rock_genre'].isin([ROCK_GENRE])]
songs = get_songs(rock)
states, songs = get_states(songs, order=ORDER)
TM, tr_pr = transition_matrix_and_heat_map(states, songs)
TM

new_chorale = generate_chords(states, 20, tr_pr, initial_state=INITIAL_STATE)
if ORDER > 1:
    new_chorale = flatten_chorale(new_chorale, ORDER)
play_music(new_chorale)



### ARTIST BASED
ARTISTS = {
    'Mayer' : '0hEurMDQu99nJRq8pTxO14', 
    'Joel' : '6zFYqv1mOsgBRQbae3JJ9e',
    'Beatles' : '3WrFJ7ztbogyGnTHbHJFl2',
}

# INPUTS
ORDER = 2
INITIAL_STATE = 'Em_G'

### Higher Order Markov Chains by Artist
artist = df[df['spotify_artist_id'].isin([ARTISTS['Mayer']])]
artist.head()
songs = get_songs(artist)
if ORDER > 1:
    artist_states, artist_songs = get_states(songs, order=ORDER)
else:
    artist_states = get_states(songs, order=ORDER)
    artist_songs = songs
TM, tr_pr = transition_matrix_and_heat_map(artist_states, artist_songs)
TM

strong_components = connected_components(tr_pr)
component_states = components_to_states(artist_states, strong_components, 10)
component_states

new_chorale = generate_chords(artist_states, 20, tr_pr, initial_state=INITIAL_STATE)
new_chorale = flatten_chorale(new_chorale, ORDER)
print(new_chorale)
play_music(new_chorale)
