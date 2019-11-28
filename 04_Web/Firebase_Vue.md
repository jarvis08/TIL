# Firebase Hosting

[Firebase Website](https://firebase.google.com/?gclid=Cj0KCQiA2vjuBRCqARIsAJL5a-LDjPT-j6KGphu4NA8Y79g45BJE85JbzwIZtXjZfdocgRJpwrei90IaAr8fEALw_wcB)

1. [Web] Firebase App 생성

2. [Local] Firebase CLI 설치

   ```bash
   $ npm install -g firebase-tools
   $ firebase login
   # Deploy를 원하는 app의 디렉토리로 이동
   $ firebase init
   ? Please select an option: Use an existing project
   ? Select a default Firebase project for this directory: ojodda-e7864 (ojodda)
   ? What do you want to use as your public directory? dist
   ? Configure as a single-page app (rewrite all urls to /index.html)? Yes
   ```

3. [Local] Compile(Build) Local App

   ```bash
   $ npm run build
   ```

4. [Local] Deploy App to Firebase

   ```bash
   $ firebase deploy
   Project Console: https://console.firebase.google.com/project/ojodda-e7864/overview
   Hosting URL: https://ojodda-e7864.firebaseapp.com
   ```

__위 코드들은 Firebase 웹사이트에서도 순서대로 지시해줍니다.__

이제 https://vue-deploy-test-1f76f.firebaseapp.com 주소로 나의 app에 접속할 수 있습니다.

