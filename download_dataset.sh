#!/bin/bash
"""
Description:
	Download Movielens dataset.

Usage:

	./download_dataset [option_name]

		option_name: download_1m,
								 download_20m,
								 download_latest,
								 download_latest_small
		default: download_20m

Author:

	johsonkee[876688461@qq.com]
"""
function download_20m {
	echo "Download ml-20m"
	curl -O http://files.grouplens.org/datasets/movielens/ml-20m.zip
}

function download_1m {
	echo "Downloading ml-1m"
	curl -O http://files.grouplens.org/datasets/movielens/ml-1m.zip
}

function download_latest_small {
	echo "Downloading ml-lastest-small"
	curl -O http://files.grouplens.org/datasets/movielens/ml-lastest-small.zip
}

function download_latest {
	echo "Downloading ml-lastest"
	curl -O http://files.grouplens.org/datasets/movielens/ml-lastest.zip
}

if [[ $1 == "" ]]
then
	download_20m
fi

if [[ $1 == "ml-1m" ]]
then
	download_1m
fi

if [[ $1 == "ml-20m" ]]
then
	download_20m
fi

if [[ $1 == "ml-latest-small" ]]
then
	download_latest_small
fi

if [[ $1 == "ml-latest" ]]
then
	download_latest
fi
