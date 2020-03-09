#!/bin/sh

for i in img/training/*.jpg
do
	echo $i
	python3 quantizer.py "$i"
done
