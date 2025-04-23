# MIDISym
MIDISym is a Python library that reads, analyses, converts any symbolic music formats (MIDI, XML, ...) into MIR-friendly format. It supports the conversion of representations such as matrix, tokens, and even graphs. It supports MIDI-level analysis such as melody extraction(skyline algorithm), feature extractions such as pitch class, pitch octave, durations. It also contains exporting features such as wav rendering.

## Installation
```bash
pip install .

```

## Usage
```
    parser = MidiParser(file_fp)
    
    # parser.sym_music_container.instruments[0].notes[0].pitch
    
    parser.dump()
```
