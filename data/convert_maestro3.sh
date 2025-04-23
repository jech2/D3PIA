#! /bin/bash
set -e

echo Converting the audio files to FLAC ...
COUNTER=0
for f in maestro-v3.0.0/*/*.wav; do
    COUNTER=$((COUNTER + 1))
    echo -ne "\rConverting ($COUNTER/1276) ..."
    ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 16000 ${f/\.wav/.flac}
done

echo
echo Preparation complete!