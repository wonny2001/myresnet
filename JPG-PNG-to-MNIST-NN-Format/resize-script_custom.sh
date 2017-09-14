#!/bin/bash

#simple script for resizing images in all class directories
#also reformats everything from whatever to png

size=28
dir=eye-images


if [ `ls "$dir"/test-images/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in "$dir"/test-images/*/*.jpg; do
    convert "$file" -resize "$size"x"$size"\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls "$dir"/test-images/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in "$dir"/test-images/*/*.png; do
    convert "$file" -resize "$size"x"$size"\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi

if [ `ls "$dir"/training-images/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in "$dir"/training-images/*/*.jpg; do
    convert "$file" -resize "$size"x"$size"\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls "$dir"/training-images/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in "$dir"/training-images/*/*.png; do
    convert "$file" -resize "$size"x"$size"\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi
