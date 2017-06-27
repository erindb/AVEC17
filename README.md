# AVEC17

## Directory structure

* `data`
	- Will never be committed to git, but contains several datasets for the project
* `models`
	- Each directory corresponds to a particular model. So far we have two models to write for Desmond's dataset. A more detailed description of the model can be found in the `README.md` at the top of that model's directory.

## git + Dropbox

**DO NOT COMMIT DATA EVER**

*these instructions will work for mac and linux, but not windows*

If you're working on git, clone [this repo](https://github.com/erindb/AVEC17) somewhere else on your computer. Make a symbolic link as the data folder into your cloned repo. Do not make changes in the Dropbox folder itself.

	ln -s ~/Dropbox/AVEC17/data .

When you push, use `./push.sh` or `sh push.sh`. This script will push to github, and then go to the Dropbox folder to pull the changes. **You might need to change the location of the Dropbox folder** in `push.sh`.

Let Erin know when (not if) this system fails.
