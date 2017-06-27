#!/bin/bash

set -e
git push
wd=`pwd`
cd ~/Dropbox/Stanford\ AVEC\ Team\ Folder/AVEC17/data
git pull
cd $wd
