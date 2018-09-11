# tf_1.8_xla_doc

## Requirement
- Doxygen
	- `sudo pacman -Sy doxygen graphviz` on Arch

## How to Verify the Result
I suggest you run `python3 -m http.server` on `/docs` directory. If you need to re-direct out from VM or something like that. I suggest you use `ngrok http 8000`

## How to Commit
Before commit please run `doxygen` on root directory to make sure the generated html is up to date.
