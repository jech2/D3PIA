from pathlib import Path
from .midi import MidiParser
from .container import SymMusicContainer
import os


class XmlParser(MidiParser):
    def __init__(
        self,
        file_path: Path | str | None = None,
        use_symusic: bool = False,
        sym_music_container: SymMusicContainer | None = None,
    ):
        # for parsing
        self.use_symusic = use_symusic
        self.init_parser()
        if file_path:
            # check file_path ext is xml
            file_path = Path(file_path) if isinstance(file_path, str) else file_path
            if file_path.suffix not in [".xml", ".musicxml"]:
                raise ValueError(
                    f"File extension must be .xml, got {Path(file_path).suffix}"
                )
            else:
                midi_path = file_path.with_suffix(".mid")
                if not midi_path.exists():
                    self.convert_xml_to_midi(file_path, midi_path)
                print(
                    "currently, the xml parser is using the converted midi file from musescore"
                )
                self.sym_music_container = self.parse(midi_path)
        elif sym_music_container:
            self.sym_music_container = sym_music_container

    def convert_xml_to_midi(self, xml_path: Path, midi_path: Path):
        cmd = f"QT_QPA_PLATFORM=offscreen mscore -o {str(midi_path)} {str(xml_path)}"

        os.system(cmd)
