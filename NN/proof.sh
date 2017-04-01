#!/bin/bash

for length in 70 100 
do
    for complexity in 5 10 20 25 
    do
        python run.py $length $complexity
    done
done
