#!/bin/bash

set -e
git push
wd=`pwd`
cd ~/Dropbox/AVEC17/
git pull
cd $wd