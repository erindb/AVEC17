# AVEC17

## git + Dropbox

**DO NOT COMMIT DATA EVER**

*these instructions will work for mac and linux, but not windows*

If you're working on git, clone [this repo](https://github.com/erindb/AVEC17) somewhere else on your computer. Copy the data folder into your cloned repo. Do not make changes in the Dropbox folder itself.

When you push, use `./push.sh` or `sh push.sh`. This script will push to github, and then go to the Dropbox folder to pull the changes. **You might need to change the location of the Dropbox folder** in `push.sh`.

Let Erin know when (not if) this system fails.
