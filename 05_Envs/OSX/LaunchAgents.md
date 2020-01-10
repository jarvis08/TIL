# 시작 프로그램 관리

일반적으로 시작 프로그램을 관리하는 것은 `시스템 환경설정`에서 `사용자 및 그룹`, `로그인 항목`에서 간편하게 할 수 있습니다. 그런데 분명 그곳에 위치하지 않는 자동으로 시작되는 프로그램들이 상당히 많습니다.

이럴 때에는 터미널을 켜고 다음 명령어를 실행하면,

```bash
$ launchctl list
PID	Status	Label
686	0	com.apple.mdworker.shared.01000000-0300-0000-0000-000000000000
330	0	com.apple.trustd.agent
-	0	com.apple.MailServiceAgent
-	0	com.apple.mdworker.mail
-	0	com.apple.appkit.xpc.ColorSampler
680	0	com.apple.mdworker.shared.06000000-0200-0000-0000-000000000000
320	0	com.apple.cfprefsd.xpc.agent
-	0	com.apple.coreimportd
```

위와 같이 결과가 출력됩니다. 여기서 원하는 응용 프로그램 이름을 검색한 후 다음 명령어를 수행하면, 시작 프로그램의 목록에서 제외됩니다.

`launchctl unload -w /Library/LaunchAgents/프로그램명.plist`

