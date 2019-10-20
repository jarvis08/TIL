# Errors

## After OS X update

```bash
$ git --version
xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun
```

### Solution

Reinstall the command line tools

출처: [Stack Over Flow](https://stackoverflow.com/questions/58280652/git-doesnt-work-on-macos-catalina-xcrun-error-invalid-active-developer-path)

```bash
$ xcode-select --install
```



