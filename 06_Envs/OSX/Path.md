# Path

## What is Path

### 1.

```
>> What is this "export" phrase at the start?

export is a command (more precisely it's a Bash builtin, i.e. it's not an executable present in PATH, it's a command that Bash has built-in in itself).

>> Is it exporting the data to be available for Bash?

export sets the environment variable on the left side of the assignment to the value on the right side of the assignment; such environment variable is visible to the process that sets it and to all the subprocesses spawned in the same environment, i.e. in this case to the Bash instance that sources ~/.profile and to all the subprocesses spawned in the same environment (which may include e.g. also other shells, which will in turn be able to access it).

>> What is the first PATH and what is the second $PATH, and why do we need two?

The first PATH as explained above is the environment variable to be set using export.

Since PATH normally contains something when ~/.profile is sourced (by default it contains /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games), simply setting PATH to ~/.composer/vendor/bin would make PATH contain only ~/.composer/vendor/bin.

So since references to a variable in a command are replaced with (or "expanded" to) the variable's value by Bash at the time of the command's evaluation, :$PATH is put at the end of the value to be assigned to PATH so that PATH ends up containing ~/.composer/vendor/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games (i.e. what PATH contains already plus ~/.composer/vendor/bin: at the start).
```

### 2.

https://help.ubuntu.com/community/EnvironmentVariables will probably help you. Also `man bash`may be very helpful with understanding how that works (at least in Bash)

Anyway - as for `PATH=` you're basically setting the `PATH` variable, adding some new paths to search through, adding at the end already / previously set paths, with `$PATH` (which is basically a reference to the `PATH` variable).

So, say your `PATH` was so far set to something like:

```bsh
PATH="x:y:z"
```

and then you set

```bsh
PATH="a:b:c:$PATH"
```

your `PATH` after that will be like:

```bsh
a:b:c:x:y:z
```

I hope that makes sense.

And on top of that you export the new variable so it's known in your environment including also child processes / subshells.

Just be aware also that the order of the directories as set in `PATH` can be important. And something like `PATH="$PATH:a:b:c"` will give you the result:

```bsh
x:y:z:a:b:c
```

which will affect the order of directories / paths while searching for a command (if you have your command in more than one of directories, the first found will be used - which may give you some unexpected results sometimes).

### 3.

Here's the command so that everybody can follow along as they go through the bullet points. `export PATH="~/.composer/vendor/bin:$PATH"`

- `export` shell built-in (meaning there is no `/bin/export` ,it's a shell thing) command basically makes environment variables available to other programs called from `bash` ( see the linked question in Extra Reading ) and the subshells.
- Assignment in shell will take expansion first , then assignment will take place second. So what is inside double quotes gets expanded first, saved to `PATH` variable later.
- `$PATH` is the default `PATH` assignment ( or at least what the variable looks like up till this command appears in your `.bashrc` or `.profile`), and expand it.
- `~/.composer/vendor/bin` is going to expand to `/home/username/.composer/vendor/bin` , where `.composer` is hidden folder due to the leading dot.
- That short `~/.composer/vendor/bin:$PATH` have now transformed into long list of folders, separated by `:`. Everything is enclosed into double quotes so that we include folders with spaces in their path.
- Finally everything is stored into `PATH` variable and external commands allowed to use it

**Simple Example**

My interactive shell is actually `mksh` , which happens to also have `export` builtin. By using `export`to set `VAR`, my variable can be passed to and used by subsequent chain of commands/subprocesses, where I exported that same variable

```bsh
$ echo $SHELL            
/bin/mksh
$ VAR="HelloAskUbuntu"
$ bash -c 'echo $VAR' 
$ export VAR="HelloAskUbuntu"                                                  
$ bash -c 'echo $VAR'                                                          
HelloAskUbuntu
$ 
```

**Extra Reading**

- [What does “export” do in shell programming?](https://stackoverflow.com/a/7411509/3701431)

---

## 공통

- 현재 설정된 Path 확인하기

  ```shell
  $ echo $PATH
  ```
  
  ```shell
  # 결과
  /usr/local/sbin:/usr/local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/Applications/Server.app/Contents/ServerRoot/usr/bin:/Applications/Server.app/Contents/ServerRoot/usr/sbin
  ```

---

## 사용자 PATH


- 저장 파일

  `~/.bash_profile`

### 설정 순서

1. PATH라는 것이 있는지 확인

2. 없을 경우 추가, 공백 주의

   `export PATH=${PATH}` 

그리고 그 뒤에 다른 Path가 필요할 경우 `{PATH}`뒤에 `:` 적어서 경로를 이어준다.

## 관리자 PATH

- `sudo nano /etc/paths`

  ```shell
  # 결과
  /usr/local/bin
  /usr/bin
  /bin
  /usr/sbin
  /sbin
  ```

- 저장 파일

  `/etc/paths`

---