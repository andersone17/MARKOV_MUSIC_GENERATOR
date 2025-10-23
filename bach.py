# %%

import pickle, music21, re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

## Load and clean data:

cols = ['song','time',
        '1','2','3','4','5','6','7','8','9','10','11','12',
        'root','what','chord']
bach = pd.read_csv('./bach.data',
                   header=None,
                   names = cols)
bach = bach.loc[:,['song','time','chord']]
bach.head(10)

seq = []
songs = bach['song'].unique()
for song in songs: # Cleans up the chord symbols for music21
    df_s = bach.loc[ bach['song']==song, :]
    lt = list(df_s['chord'])
    lt = [ re.sub(r' ','', ch) for ch in lt]
    lt = [ re.sub(r'_','', ch) for ch in lt]
    lt = [ re.sub(r'M','', ch) for ch in lt]
    lt = [ re.sub(r'b','-', ch) for ch in lt]
    lt = [ re.sub(r'd','dim', ch) for ch in lt]
    lt = [ re.sub(r'dim6','dim', ch) for ch in lt]
    lt = [ re.sub(r'm4','m', ch) for ch in lt]
    seq.append( lt )

bach = seq



# %%

## Determine state space:
states = set()
for song in bach:
    states = states.union( set(song) )
states = list(states)

print('States:\n', np.array(states) )


# %%

## Play the state space
stream = music21.stream.Stream()
for chord_symbol in states:
    chord = music21.harmony.ChordSymbol(chord_symbol)
    chord.duration = music21.duration.Duration(2.0)  
    stream.append(chord)
stream.write('midi', fp='music.mid')
stream.show('midi')  

# An authentic Bach chorale
stream = music21.stream.Stream()
for chord_symbol in bach[5]:
    chord = music21.harmony.ChordSymbol(chord_symbol)
    chord.duration = music21.duration.Duration(2.0)  
    stream.append(chord)
stream.write('midi', fp='music.mid')
stream.show('midi')  


# %%

## Create a S X S transition matrix, and find the transition counts:

S = len(states)
T = len(bach)
tr_counts = np.zeros( (S, S) )

## Compute transition counts:
for song in bach:
    seq = np.array(song)
    for t in range(1,len(seq)):
        # Current and next tokens:
        x_tm1 = seq[t-1] # previous state
        x_t = seq[t] # current state
        # Determine transition indices:
        index_from = states.index(x_tm1)
        index_to = states.index(x_t)
        # Update transition counts:
        tr_counts[index_to, index_from] += 1

print('Transition Counts:\n', tr_counts)

# Sum the transition counts by row:
sums = tr_counts.sum(axis=1, keepdims=True)
print('State proportions: \n')

# Normalize the transition count matrix to get proportions:
tr_pr = np.divide(tr_counts, sums, 
                             out=np.zeros_like(tr_counts), 
                             where=sums!=0)

print('Transition Proportions:\n')

tr_df = pd.DataFrame(np.round(tr_pr,2), index=states, columns=states)
print(tr_df)

plt.figure(figsize=(12, 10))
sns.heatmap(tr_pr, 
            cmap='Blues',    
            square=True,          
            xticklabels=states,
            yticklabels=states,
            cbar_kws={'label': 'Transition Probability'})

plt.title('Transition Probabilities')
plt.xlabel('...To State')
plt.ylabel('From State...')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()





# %%

## Random chords:
random_pr = row_sums/np.sum(row_sums)

np.random.seed(100) 
initial_state = np.random.choice(states) # Choose an initial state at random
state_index = states.index(initial_state) # Get the index of the initial state
print(f'Initial state: {initial_state}') 
n_sim = 20

simulation = [initial_state]
for t in range(n_sim-1): 
    state_index = np.random.choice(len(states), p=random_pr) # Choose new state index
    simulation.append(states[state_index]) # Append new state to simulation
print(simulation)

new_chorale = [state.split()[-1] for state in simulation] # Convert to chords

# New random chorale
stream = music21.stream.Stream()
for chord_symbol in new_chorale:
    chord = music21.harmony.ChordSymbol(chord_symbol)
    chord.duration = music21.duration.Duration(2.0)  
    stream.append(chord)
stream.write('midi', fp='music.mid')
stream.show('midi')  





# %%

G = nx.from_numpy_array(tr_pr, create_using=nx.DiGraph())

# Check if strongly connected (every chord can reach every other chord)
is_strongly_connected = nx.is_strongly_connected(G)
print(f"Connected: {is_strongly_connected}")

# Get connected components if not connected
if not is_strongly_connected:
    strong_components = list(nx.strongly_connected_components(G))
    print(f"Number of strongly connected components: {len(strong_components)}")




# %%

## Order 2 Markov transitions:

order = 2
songs = []
for song in bach:
    T = len(song)
    entry = [' '.join(song[(t-order-1):(t-1)]) for t in range(order+1, T)]
    songs.append(entry) 

states = set()
for song in songs:
    states = states.union( set(song) )
states = list(states)

print('States:\n', np.array(states) )

S = len(states)
tr_counts = np.zeros( (S, S) )

# Fix the transition counting (around line where you have the nested loops):
for song in songs:
    seq = np.array(song)
    for t in range(1,len(seq)):
        x_tm1 = seq[t-1] # previous state
        x_t = seq[t] # current state
        index_from = states.index(x_tm1)
        index_to = states.index(x_t)
        tr_counts[index_to, index_from] += 1  # TO, FROM not FROM, TO

sums = tr_counts.sum(axis=1, keepdims=True)
state_props = sums/np.sum(sums)

# And fix the transition probability normalization:
tr_pr = np.divide(tr_counts, sums, 
                 out=np.zeros_like(tr_counts), 
                 where=sums!=0)
print('\nTransition Proportions:\n', tr_pr)


# %%

plt.figure(figsize=(12, 10))
sns.heatmap(tr_pr, 
            cmap='Blues',    
            square=True,          
            xticklabels=states,
            yticklabels=states,
            cbar_kws={'label': 'Transition Probability'})

plt.title('Transition Probabilities')
plt.xlabel('...To State')
plt.ylabel('From State...')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# %%

## Connected?

G = nx.from_numpy_array(tr_pr, create_using=nx.DiGraph())

# Check if strongly connected (every chord can reach every other chord)
is_strongly_connected = nx.is_strongly_connected(G)
print(f"Strongly connected: {is_strongly_connected}")

# Check if weakly connected (ignoring edge direction)
is_weakly_connected = nx.is_weakly_connected(G)
print(f"Weakly connected: {is_weakly_connected}")

# Get connected components if not connected
if not is_strongly_connected:
    strong_components = list(nx.strongly_connected_components(G))
    print(f"Number of strongly connected components: {len(strong_components)}")


# %%

## Generate new music:

np.random.seed(10000) # Favorite
#np.random.seed(5000) 
initial_state = np.random.choice(states) # Choose an initial state at random


initial_state = 'E B'
#initial_state = 'Am Dm'


state_index = states.index(initial_state) # Get the index of the initial state
print(f'Initial state: {initial_state}') 

n_sim = 20

simulation = [initial_state]
for t in range(n_sim-1): 
    pr_t = tr_pr[:,state_index] # Transition probabilities at this state
    state_index = np.random.choice(len(states), p=pr_t) # Choose new state index
    simulation.append(states[state_index]) # Append new state to simulation

print(simulation)

new_chorale = [state.split()[-1] for state in simulation] # Convert to chords

# New Bach chorale
stream = music21.stream.Stream()
for chord_symbol in new_chorale:
    chord = music21.harmony.ChordSymbol(chord_symbol)
    chord.duration = music21.duration.Duration(2.0)  
    stream.append(chord)
stream.write('midi', fp='music.mid')
stream.show('midi')  

# %%
