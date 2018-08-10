#!/bin/bash
# runs a small dataset in personal computer
# to use the script:
#   run_and_time.sh <random seed 1-5>

THRESHOLD=0.6289
BASEDIR=$(dirname -- "$0")

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Get command line seed
seed=${1:-1}

echo "unzip ml-latest-small.zip"
if unzip ml-latest-small.zip
then
    echo "Start processing ml-latest-small/ratings.csv"
    t0=$(date +%s)
	python3 $BASEDIR/convert.py ml-latest-small/ratings.csv ml-latest-small --negatives 999
    t1=$(date +%s)
	delta=$(( $t1 - $t0 ))
    echo "Finish processing ml-latest-small/ratings.csv in $delta seconds"

    echo "Start training"
    t0=$(date +%s)
	python3 $BASEDIR/ncf.py ml-latest-small -l 0.0005 -b 2048 --layers 256 128 64 -f 64 \
		--seed $seed --threshold $THRESHOLD --processes 1
    t1=$(date +%s)
	delta=$(( $t1 - $t0 ))
    echo "Finish training in $delta seconds"

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="recommendation"


	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Problem unzipping ml-latest-small.zip"
	echo "Please run 'download_data.sh && verify_datset.sh' first"
fi
