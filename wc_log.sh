#! /bin/bash

src=$1
log=$2
#freq=2
#freq = 60 #A minute
#freq=1800 #Half an hour
freq=600 #10 minutes
#freq = 3600 #An hour

function count {
	cnt=$(wc -w $src)
	time=$(date)
	echo $time, $cnt >> $log
}

while :; do count; sleep $freq; done;
