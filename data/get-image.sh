#!/bin/baSH

for i in  {106000..107000}
do
	wget https://wallpaperaccess.com/full/$i.jpg
	mv *.jpg training/
done
