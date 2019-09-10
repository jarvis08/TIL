# .gitignore

[gitignore.io](http://gitignore.io/) Framework, OS, IDE 등에 맞는 gitignore 목록 검색

- git에 upload하고 싶지 않은 대상을 `.gitignore`에 작성,

  해당 파일에 작성해 둔 파일들은 git 활동에서 자동으로 제외

  - 자신의 directory를 기준으로, 모든 하위 directory까지 영향

  - `파일명/디렉토리명` : file, directory 제외
  - `디렉토리명/*` : directory 내용물을 모두 제외

  ```shell
  # jupyer notebook checkpoints 제거
  $ code .gitignore
  >> 내부 작성 .ipynb_checkpoints
  
  # .git 폴더 제거
  $ rm -rf .git/
  
  # 현재 디렉토리의 .DS_Store 제거
  $ git rm --cached .DS_Store
  # 모든 remote git repository에서 .DS_Store 찾아서 삭제
  find . -name .DS_Store -print0 | xargs -0 git rm --ignore-unmatch
  # 앞으로의 모든 repository에서의 .DS_Store upload 예방
  # 1. global하게 사용할 .gitignore를 어딘가에 생성.
  # 파일 예시
  echo .DS_Store >> ~/.gitignore_global
  # 2. git에게 모든 repository에 사용할 것을 선언하기
  git config --global core.excludesfile ~/.gitignore_global
  # 참고자료 - https://stackoverflow.com/questions/18393498/gitignore-all-the-ds-store-files-in-every-folder-and-subfolder
  
  # TIL/50_SSAFY/8ython directory 내부에 .git이 따로 있었으나, 제외하고 다시 TIL의 .git으로 포함시킴
  $ git rm -rf ~/TIL/50_SSAFY/8ython.git
  $ git rm --cached 8ython/
  검색창 :: gitignore.io/api/python
  ```