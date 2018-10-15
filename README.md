# tf_1.8_xla_doc

## Requirement
- Doxygen
	- `sudo pacman -Syu doxygen graphviz` on Arch

## How to Write The Document
Because `EXTRACT_ALL` is set to `NO`. So it requires you to have every parent classes/namespaces(except anonymous namespace) documented at least once(can be content free but must have `/**\n [gibberish]\n */`). Otherwise the function would not show up correctly on HTML pages.

Documentation that only contains `\brief` directive will not show up in "Function Documentation"

Remember to add `/** \file\n*/` to the top of file you want it show up in the HTML

Use `/** \todo something_to_do */` to mark the description in a block. You can view all the "TODO" in "Related Pages"

## How to Verify The Result
I suggest you run `python3 -m http.server` on `/docs` directory. If you need to re-direct out from VM or something like that. I suggest you use `ngrok http 8000`

## Auto Reload
If you want to let `doxygen` reload each time you change a file you can do the following:
1. `sudo pip3 install watchdog`
2. `watchmedo shell-command --rursive --command='doxygen' tensorflow/tensorflow/`

## Known Issues
- Seems like doxygen cannot generate call/caller graph for same function name with different parameters and return type(i.e. not function overloading)

## How to Commit
Before commit please run `doxygen` on root directory to make sure the generated html is up to date.
