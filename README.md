# tf_1.8_xla_doc

## Requirement
- Doxygen
	- `sudo pacman -Sy doxygen graphviz` on Arch

## How to Verify the Result
I suggest you run `python3 -m http.server` on `/docs` directory. If you need to re-direct out from VM or something like that. I suggest you use `ngrok http 8000`

### Auto Reload
If you want to let `doxygen` reload each time you change a file you can do the following:
1. `sudo pip3 install watchdog`
2. `watchmedo shell-command --rursive --command='doxygen' tensorflow/tensorflow/`

## How to Commit
Before commit please run `doxygen` on root directory to make sure the generated html is up to date.
