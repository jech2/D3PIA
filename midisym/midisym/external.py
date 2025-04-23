# source code from muspy: https://muspy.readthedocs.io/en/stable/_modules/muspy/external.html#download_musescore_soundfont
import urllib.request
from pathlib import Path
from midisym.parser.midi import MidiParser

__all__ = [
    "download_musescore_soundfont",
    "get_musescore_soundfont_dir",
    "get_musescore_soundfont_path",
    "read_midi",
]

def read_midi(midi_path: str, **kwargs) -> MidiParser:
    return MidiParser(midi_path, **kwargs)


def get_musescore_soundfont_dir() -> Path:
    """Return path to the MuseScore General soundfont directory."""
    return Path.home() / ".muspy/musescore-general"


def get_musescore_soundfont_path() -> Path:
    """Return path to the MuseScore General soundfont."""
    return get_musescore_soundfont_dir() / "MuseScore_General.sf3"


def download_musescore_soundfont(overwrite: bool = False):
    """Download the MuseScore General soundfont.

    Parameters
    ----------
    overwrite : bool, default: False
        Whether to overwrite an existing file.

    """
    if not overwrite and get_musescore_soundfont_path().is_file():
        print("Skip downloading as the MuseScore General soundfont is found.")
        return

    # Make sure the directory exists
    get_musescore_soundfont_dir().mkdir(parents=True, exist_ok=True)

    # Download the soundfont
    print("Start downloading MuseScore General soundfont.")
    prefix = "ftp://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/"
    urllib.request.urlretrieve(
        prefix + "MuseScore_General.sf3", get_musescore_soundfont_path()
    )
    print(
        "MuseScore General soundfont has successfully been downloaded to : "
        "{}.".format(get_musescore_soundfont_dir())
    )

    # Download the license
    urllib.request.urlretrieve(
        prefix + "MuseScore_General_License.md",
        get_musescore_soundfont_dir() / "MuseScore_General_License.md",
    )
