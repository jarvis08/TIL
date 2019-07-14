# SSAFY_DAY_4

- Static Page

  : 누구에게나 동일한 결과 제공

  github을 통해 제공하기 쉬우며, 현재 여러 회사들의 기술 블로그 또한 이를 이용

- Dynamic Page

  : 사용자의 요청에 따라 동적으로 변화시켜 제공

  flask에서 dynamic 디렉토리를 설정하는 이유

- Tech Crunch Berlin

  세계적인 기술 컨퍼런스

---

- Fake Google 만들기

  `https://www.google.com/search?q=` : 해당 문장 뒤에 검색어를 입력하면 검색이 가능

  따라서 `<form action="https://www.google.com/search">` 을 구성한 이후

  `<input name="q">` 을 통해 'q' 파라미터를 지정하여 검색 정보 전달

