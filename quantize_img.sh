#!/bin/sh

for i in img/test/*.jpg
do
	echo $i
	python3 quantizer.py "$i"
done
