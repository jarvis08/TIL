# WEB Service

참고 자료 : ./10_Lecture_Scripts/01_intro_to_web.pdf

<br>

### World Wide Web, WWW, W3

인터넷에 연결된, 컴퓨터를 통해, 사람들이 정보를 공유할 수 있는 전 세계적인 정보 공간

최초의 웹은 Next 회사의 OS Nexus에서 구동

W3C(공기업)와 WHATWG(사기업)는 2019년 표준을 통합하는데에 합의

- 요청, request
  - GET

    해당 Data를 달라고 요청

    url 뒤쪽에 params를 이용하여 보내기도 함

    HTML로 작성된 문서파일 하나를 받을 수 있다.

  - POST

    Data를 보내면서 처리해 줄 것을 요청
  
- 응답, response

  요청 발생 시 Server에서 응답 발송

- Web Service 만들기

  = Server에서 요청과 응답을 처리할 프로그램을 개발한다

<br>

### Static Web

단순한 웹 서비스로, 정적이며 정해진 응답만 가능

요청이 단순할 것을 가정하며, 어떤 요청이든 동일한 답변만 가능

e.g., Blogs, Portfolio page, Github Pages

<br>

### Web Browser

클라이언트가 요청을 보내는 프로그램(수단)

한국의 경우, 2016년 기준 Chrome이 IE를 역전

- 왜 Chrome이 선호되고 있나

  Open Source의 세상인 현재, 표준화(W3C 지정)가 중요한 항목.

  과거 시장을 장악한 MS(IE)는 표준을 지키지 않는 활동을 많이 진행했고,

  현재 표준에 맞춰진 코드들이 잘 돌아가지 않는 상황이 도래.

- Web을 사용한 접근

  - 내컴/dir_1/dir_2/filename.pdf

  - 남의컴/dir_1/dir_2/filename.pdf

    남의컴 주소 = 172.217.27.78

    - Domain 구매

      Domain Name System, DNS.

      전화번호부 처럼, Entity와 실제 주소를 매칭

      e.g., naver.com / google.com

      DNS Server가 중단되면 직접 주소를 입력해야 접속 가능

  - 사용자가 `Daum.net`을 주소창에 치면 발생하는 일

    1. Request URL을 DNS Server에 전송
    2. DNS Server는 원래 Daum의 주소를 브라우저에 전송
    3. Browser는 그 주소를 이용하여 Daum에 접속

  - 만약 Browser에서 F12, Network tab의 Remote Address를 통해 접속 시도 시,

    Proxing을 방지하기 위해 이러한 직접 접근을 Browser에서 차단

    - Proxy

      Client와 Server의 중간에서 정보를 빼내는 해킹 방법

<br>

### url이 길고 길어지는 이유

사용자의 footstep을 기록 및 tagging하여 데이터화

- `장바구니`와 같은 컨텐츠는 `cookie`에 기록

  `cookie`는 browser를 통해 저장

- app 간의 이동에는 `cookie`를 활용할 수 없으며, 이는 url tagging이 대신

  기업은 tagging된 데이터를 활용하여 사용자가 어떤 route를 통해 서비스를 이용하는지 확인 가능