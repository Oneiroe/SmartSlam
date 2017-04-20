#! /bin/sh

# First variable: audio capturing device hardware number
HW="$1"

# Second Variable: sampling duration in seconds
DURATION="$2"

# Third variable: folder PATH where to store the recording
DIR="$3"
cd ${DIR}

BASE=sample
i=0
if [ -e ${BASE}-${i}.wav ] ; then
    i=1
    while [ -e ${BASE}-${i}.wav ] ; do
        i=$((i+1))
    done
fi
NAME=${BASE}-${i}

do_record () {
    # record
    arecord -f cd -r 48000 -D hw:${HW} -d "$1" ${DIR}${NAME}.wav
}

do_stream () {
    # stream to audio output
    arecord -D plughw:1,0 -f dat | aplay -f dat
}

do_label() {
    # add default label to last recorded file
    echo "${NAME},default">>${DIR}labels.csv
}

do_record ${DURATION}
do_label