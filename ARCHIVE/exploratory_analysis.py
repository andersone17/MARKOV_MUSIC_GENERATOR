# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from music21 import harmony

# Initial File Read
chords = pd.read_csv(r"DATA\chordonomicon_v2.csv")


# INPUTS
GENRE = 'rock'
ARTIST_ID = "0hEurMDQu99nJRq8pTxO14"

# Subsetting
subset = chords[chords['spotify_artist_id'] == ARTIST_ID]

# Generate State Space
songs = subset['chords']
cleaned_songs = [re.sub(r'<[^>]+>', '', song).strip() for song in songs]
cleaned_songs = [' '.join(song.split()) for song in cleaned_songs]
songs = [song.split() for song in cleaned_songs]

states = set()
for song in songs:
    new_songs = set(song)
    states = states.union(new_songs)
states = list(states)
len(states)

# Create Transition Matrix
S = len(states)
transitions = np.zeros((S,S))
for song in songs:
    for i in range(1, len(song)):
        current_chord = song[i]
        previous_chord = song[i-1]
        current_idx = states.index(current_chord)
        prev_idx = states.index(previous_chord)
        transitions[current_idx, prev_idx] +=1 
        

sums = transitions.sum(axis=0, keepdims=True)
transition_prob = np.divide(
    transitions, sums, 
    out=np.zeros_like(transitions), 
    where=sums!=0
)
transition_matrix = pd.DataFrame(transition_prob, index=states, columns=states)
transition_matrix.style.background_gradient(cmap='Blues')

# Show Heatmap
plt.figure(figsize=(13, 11))
sns.heatmap(transition_matrix, 
            cmap='Blues',
            square=True,          
            xticklabels=states,
            yticklabels=states,
            cbar_kws={'label': 'Transition Probability'})
plt.title('Transition Probabilities')
plt.xlabel('...To State')
plt.ylabel('From State...')
plt.show()


# Clean the CHords so that Music 21 Can read them
# Function to clean chord names so they are recognized by music21
def clean_chords(chords):
    cleaned = []
    for chord_symbol in chords:
        # Replace 'smin' with '#min'
        chord_symbol = chord_symbol.replace('smin', '#min')
        # Replace 'no3d' with '5'
        chord_symbol = chord_symbol.replace('no3d', '5')
        cleaned.append(chord_symbol)
    return cleaned


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


successful_maps = {}
failed_maps = {}
for original_str in states:
    normalized_str = normalize_chord_string(original_str)
    try:
        # Try to parse the normalized chord using music21
        c = harmony.ChordSymbol(normalized_str)
        standardized_figure = c.figure
        successful_maps[original_str] = {
            'normalized': normalized_str,
            'music21_figure': standardized_figure,
        }          
    except Exception as e:
        failed_maps[original_str] = {
            'normalized': normalized_str,
            'error': str(e)
        }
print(failed_maps)


states = [normalize_chord_string(s) for s in states]


import music21

## Random chords:
np.random.seed(100) 
initial_state = np.random.choice(states) # Choose an initial state at random
state_index = states.index(initial_state) # Get the index of the initial state
print(f'Initial state: {initial_state}') 

n_sim = 20

simulation = [initial_state]
for t in range(n_sim-1): 
    pr_t = transition_prob[:,state_index] # Transition probabilities at this state
    state_index = np.random.choice(len(states), p=pr_t) # Choose new state index
    simulation.append(states[state_index]) # Append new state to simulation

new_chorale = [state.split()[-1] for state in simulation] # Convert to chords

new_chorale = clean_chords(new_chorale)

print(new_chorale)

# New random chorale
stream = music21.stream.Stream()
for chord_symbol in new_chorale:
    chord = music21.harmony.ChordSymbol(chord_symbol)
    chord.duration = music21.duration.Duration(2.0)  
    stream.append(chord)
stream.write('midi', fp='music.mid')
stream.show('midi')  