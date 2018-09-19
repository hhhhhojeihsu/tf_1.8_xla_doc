# tf_1.8_xla_doc

## Requirement
- Doxygen
	- `sudo pacman -Sy doxygen graphviz` on Arch

## Branch
There are two branch in total
- `master`: Only `/docs` directory, used to host the website on github.io
- `code`: Only `/tensorflow` directory, used to store the source code

## How to Write The Document
Because `EXTRACT_ALL` is set to `NO`. So it requires you to have every parent classes/namespaces(except anonymous namespace) documented at least once(can be content free but must have `/**\n [gibberish]\n */`). Otherwise the function would not show up correctly on HTML pages.

Documentation that only contains `\brief` directive will not show up in "Function Documentation"

Remember to add `/** \file\n*/` to the top of file you want it show up in the HTML

## How to Verify The Result
I suggest you run `python3 -m http.server` on `/docs` directory. If you need to re-direct out from VM or something like that. I suggest you use `ngrok http 8000`

### Auto Reload
If you want to let `doxygen` reload each time you change a file you can do the following:
1. `sudo pip3 install watchdog`
2. `watchmedo shell-command --rursive --command='doxygen' tensorflow/tensorflow/`

## How to Commit
Please change to `code` branch before modify source code and run `doxygen` on this branch. `.gitignore` will auto ignore `/docs` folder so you can commit safely.

If you want to publish the newest content onto github.io. Then you should switch to `master` branch and then commit the changes in `/docs` and push to origin.
