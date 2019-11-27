# Firebase for Vue.js

[Firebase Website](https://firebase.google.com/?gclid=Cj0KCQiA2vjuBRCqARIsAJL5a-LDjPT-j6KGphu4NA8Y79g45BJE85JbzwIZtXjZfdocgRJpwrei90IaAr8fEALw_wcB)

1. [Web] Firebase App 생성

2. [Local] Firebase CLI 설치

   ```bash
   $ npm install -g firebase-tools
   $ firebase login
   # Deploy를 원하는 app의 디렉토리로 이동
   $ firebase init
   ```

3. [Local] Compile(Build) Local App

   ```bash
   $ npm run build
   ```

4. [Local] Deploy App to Firebase

   ```bash
   $ firebase deploy
   Project Console: https://console.firebase.google.com/project/vue-deploy-test-1f76f/overview
   Hosting URL: https://vue-deploy-test-1f76f.firebaseapp.com
   ```

__위 코드들은 Firebase 웹사이트에서도 순서대로 지시해줍니다.__

이제 https://vue-deploy-test-1f76f.firebaseapp.com 주소로 나의 app에 접속할 수 있습니다.

