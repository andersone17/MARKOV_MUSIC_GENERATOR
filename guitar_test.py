import music21
import os

# -----------------------------------------
# 1. Create Score and Guitar Part
# -----------------------------------------
score = music21.stream.Score()
guitar_part = music21.stream.Part()
guitar_part.insert(0, music21.instrument.AcousticGuitar())

# -----------------------------------------
# 2. Define Chord Progression
# -----------------------------------------
chords = ["C", "G", "Am", "F"]
chord_duration = 4.0  # 4 beats per chord

# -----------------------------------------
# 3. Add Strummed Chords
# -----------------------------------------
for symbol in chords:
    # Create a ChordSymbol and realize its pitches
    chord_symbol = music21.harmony.ChordSymbol(symbol)
    realized = chord_symbol.figure  # for printing
    chord_symbol.duration = music21.duration.Duration(chord_duration)

    # Build the chord notes directly from the ChordSymbol’s pitches
    chord_notes = chord_symbol.pitches

    # Make a voice to hold staggered (strummed) notes
    strum = music21.stream.Voice()
    delay = 0.05  # small delay between notes for strumming

    for i, pitch in enumerate(chord_notes):
        n = music21.note.Note(pitch)
        n.quarterLength = chord_duration - i * 0.02
        strum.insert(i * delay, n)

    # Add the strummed chord to the guitar part
    guitar_part.append(strum)

# -----------------------------------------
# 4. Combine into Score and Export to MIDI
# -----------------------------------------
score.append(guitar_part)
mf = music21.midi.translate.music21ObjectToMidiFile(score)

# Force Acoustic Guitar patch (General MIDI 25 = index 24)
for track in mf.tracks:
    event = music21.midi.MidiEvent(track, type='PROGRAM_CHANGE')
    event.channel = 0
    event.time = 0
    event.data = [24]  # Acoustic Guitar (nylon)
    track.events.insert(0, event)

# -----------------------------------------
# 5. Write and Auto-Open
# -----------------------------------------
midi_path = os.path.abspath("strummed_guitar_progression.mid")
mf.open(midi_path, 'wb')
mf.write()
mf.close()

print(f"✅ Strummed guitar progression written to: {midi_path}")
os.startfile(midi_path)
