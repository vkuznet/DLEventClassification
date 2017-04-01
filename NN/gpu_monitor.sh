#!/bin/bash

gpustat -cpu --no-color | grep -o -E '.{0,13} MB'

