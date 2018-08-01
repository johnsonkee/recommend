#!/bin/bash
# runs benchmark and reports time to convergence
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

echo "unzip ml-20m.zip"
if unzip ml-1m.zip
then
#    echo "Start processing ml-20m/ratings.csv"
#    t0=$(date +%s)
#	python3 $BASEDIR/convert.py ml-1m/ratings.csv ml-1m --negatives 999
#    t1=$(date +%s)
#	delta=$(( $t1 - $t0 ))
#    echo "Finish processing ml-1m/ratings.csv in $delta seconds"

    echo "Start training"
    t0=$(date +%s)
	python3 $BASEDIR/ncf.py ml-1m -l 0.0005 -b 512 --layers 256 128 64 -f 64 \
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
	echo "Problem unzipping ml-1m.zip"
	echo "Please run 'download_data.sh && verify_datset.sh' first"
fi
