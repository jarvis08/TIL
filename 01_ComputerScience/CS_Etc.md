# Etc

---

## MVP (Minimum Viable Product)

: 최소 기능 제품

가장 필요한 기능을 빠르게 구현하여 출시한 후, 계속해서 사용자의 니드를 보완

반대 개념 - 완결성 있는 제품을 장기간에 걸쳐 개발하여 출시

---

## HASH

- `sha256` : Web의 보안 Hash 생성 방법
  - 2**256의 경우의 수로 랜덤 생성
- git log를 통해 hash를 확인 가능(sha256 급의 보안은 아님!)
  - hash로 인해 git 내용의 철자 하나만 바뀌어도 즉각적인 변화 포착

---

## RFP List, 명세서

- 제안 요청서(RFP, request for proposal)

- 요구사항 명세서(SRS, Software Requirements Specification)
  - 기능 명세 - FUNCTIONAL SPECIFICATION

    완전히 사용자 관점에서 제품이 어떻게 동작할지를 기술

    기능에 대해 이야기하고, 화면, 메뉴, 대화상자와 같은 사용자 인터페이스 부품을 명세

  - 기술 명세 - TECHNICAL SPECIFICATION

    프로그램 내부 구현을 기술

    자료구조와 관계형 데이터베이스 모델과 프로그래밍 언어, 도구, 알고리즘 선택과 같은 항목

---

## Virtual Environment

```shell
mkdir python-verualenv
# -m : modul
python -m venv ~/python-vertualenv/3.7.3
cd work-directory
# 가상환경 실행
source ~/python-virtualenv/3.7.3/Scripts/activate
# mac os
source ~/python-virtualenv/3.7.3/bin/activate
# 가상환경 종료
deactivate

# alias 이용하여 실행 코드 줄이기
# .bashrc 추가, mac OS의 경우 ~/.bash_profile
alias venv='source ~/python-virtualenv/3.7.3/Scripts/activate'
source .bashrc
source .bash_profile
```

---

## OS X

- Python Code 내에 한글을 작성하려면,

  OS X의 경우 다음 line을 기입해 넣어야 한다.

  ```python
  #-*-coding: utf-8
  ```

  