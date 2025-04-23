# original code from the library muspy: https://muspy.readthedocs.io/en/stable/_modules/muspy/outputs/audio.html#write_audio
"""Audio output interface."""
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from ..external import get_musescore_soundfont_path, download_musescore_soundfont
from .midi import MidiParser
from typing import Union


def _check_soundfont(soundfont_path):
    if soundfont_path is None:
        soundfont_path = get_musescore_soundfont_path()
    else:
        soundfont_path = Path(soundfont_path)
    if not soundfont_path.exists():
        print("Soundfont not found. Downloading MuseScore General soundfont.")
        download_musescore_soundfont()
        # raise RuntimeError(
        #     "Soundfont not found. Please download it by "
        #     "`muspy.download_musescore_soundfont()`."
        # )
    return soundfont_path

def write_audio(
    parser: MidiParser,
    path: Union[str, Path],
    audio_format: str = None,
    soundfont_path: Union[str, Path] = None,
    rate: int = 44100,
    gain: float = None,
):
    """Write a Music object to an audio file.

    Supported formats include WAV, AIFF, FLAC and OGA.

    Parameters
    ----------
    path : str or Path
        Path to write the audio file.
    music : :class:`muspy.Music`
        Music object to write.
    audio_format : str, {'wav', 'aiff', 'flac', 'oga'}, optional
        File format to write. Defaults to infer from the extension.
    soundfont_path : str or Path, optional
        Path to the soundfount file. Defaults to the path to the
        downloaded MuseScore General soundfont.
    rate : int, default: 44100
        Sample rate (in samples per sec).
    gain : float, optional
        Master gain (`-g` option) for Fluidsynth. Defaults to 1/n,
        where n is the number of tracks. This can be used to prevent
        distortions caused by clipping.

    """
    if audio_format is None:
        audio_format = "auto"
    if gain is None:
        gain = 1 / len(parser.sym_music_container.instruments)

    # Check soundfont
    soundfont_path = _check_soundfont(soundfont_path)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:

        # Write the Music object to a temporary MIDI file
        midi_path = Path(temp_dir) / "temp.mid"
        parser.dump(midi_path)

        # Synthesize the MIDI file using fluidsynth
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                "-F",
                str(path),
                "-T",
                audio_format,
                "-r",
                str(rate),
                "-g",
                str(gain),
                str(soundfont_path),
                str(midi_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
