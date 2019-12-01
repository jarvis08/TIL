## 준비물[Permalink](https://jekyllrb-ko.github.io/docs/installation/#requirements)

시작하기 전에, 자신의 시스템에 다음 것들이 준비되어 있는지 확인하세요:

- [루비](https://www.ruby-lang.org/en/downloads/) 버전 2.2.5 또는 그 이상. 모든 개발환경 헤더 포함 (루비 설치정보는 `ruby -v` 로 확인할 수 있습니다)
- [RubyGems](https://rubygems.org/pages/download) (명령어 `gem -v` 로 확인할 수 있습니다)
- [GCC](https://gcc.gnu.org/install/) 와 [Make](https://www.gnu.org/software/make/) (명령행 인터페이스에서 `gcc -v` 와 `g++ -v`, `make -v` 로 확인할 수 있습니다)

<br>

<br>

## 맥OS 에 설치

여기서는 루비 2.3.3 이 함께 제공되는 맥OS High Sierra 10.13 를 기준으로 설명하며, 이전 버전의 시스템에서는 [Homebrew 로 상위버전 루비를 설치](https://jekyllrb-ko.github.io/docs/installation/#homebrew)할 필요가 있습니다.

먼저, Native 확장기능을 컴파일할 수 있게 해주는 명령줄 도구를 설치해야 하므로, 터미널을 열어 다음 명령을 실행합니다:

```
xcode-select --install
```

<br>

### OS 의 루비 환경설정

가지고 있는 루비 버전이 요구조건을 충족하는지 확인하세요:

```
ruby -v
2.3.3
```

좋습니다. 이제 Jekyll 을 설치합시다. 또 [Bundler](https://bundler.io/) 도 필요한데, [플러그인](https://jekyllrb-ko.github.io/docs/plugins)과 [테마](https://jekyllrb-ko.github.io/docs/themes)를 사용하기 위해 필요합니다:

```
gem install bundler jekyll
```

끝났습니다. 이제 바로 사용할 수 있어요. `jekyll new jekyll-website` 로 [기본 블로그 테마](https://github.com/jekyll/minima)를 설치하거나 완전 처음부터 하나하나 시작할 수도 있습니다:

```
mkdir jekyll-website
cd jekyll-website

# Gemfile 생성
bundle init

# Jekyll 추가
bundle add jekyll

# 루비 젬 설치
bundle install
```

좋습니다. 이제 [테마](https://jekyllrb-ko.github.io/docs/themes/)를 사용하거나 [자신만의 레이아웃을 생성](https://jekyllrb-ko.github.io/docs/templates/)할 수 있습니다.

### Homebrew 로 상위버전 루비 설치하기

빌드 속도가 더 빠른 최신 버전의 루비를 설치하고자 한다면, 편리한 맥OS 용 패키지 관리자인 [Homebrew](https://brew.sh/) 를 사용하길 권합니다.

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install ruby
ruby -v
ruby 2.5.1p57 (2018-03-29 revision 63029) [x86_64-darwin17]
```

야호! 시스템에 아주 반짝반짝한 루비가 준비되었네요!

<br>

### rbenv 로 여러 버전의 루비 설치하기

개발자들은 여러 버전의 루비를 관리하기 위해 보통 [rbenv](https://github.com/rbenv/rbenv) 를 사용합니다. 하나 예를 들어보면, [GitHub Pages](https://pages.github.com/versions/) 나 [Netlify](https://www.netlify.com/docs/#ruby) 에서 사용되는 루비와 동일한 버전을 사용하고자 하는 경우 이게 아주 유용합니다.

```
# rbenv 와 ruby-build 설치
brew install rbenv

# 자신의 쉘 환경에 rbenv 가 연동되도록 설정
rbenv init

# 설치결과 검사
curl -fsSL https://github.com/rbenv/rbenv-installer/raw/master/bin/rbenv-doctor | bash
```

터미널을 재시작하면 변경사항이 적용됩니다. 이제 원하는 버전의 루비를 설치할 수 있습니다. 아래처럼 루비 2.5.1 로 시작해봅시다:

```
rbenv install 2.5.1
rbenv global 2.5.1
ruby -v
ruby 2.5.1p57 (2018-03-29 revision 63029) [x86_64-darwin17]
```

끝입니다! [rbenv 명령어 참고서](https://github.com/rbenv/rbenv#command-reference)를 한 번 잘 읽어보면 다양한 버전의 루비를 어떻게 프로젝트 별로 관리하는지 알 수 있습니다.

<br>

<br>

## 업그레이드

Jekyll 개발에 참여하기 전에, 자신이 현재 최신 버전을 사용하고 있는지 확인하고 싶을 수도 있습니다. 현재 설치된 Jekyll 의 버전을 확인하려면, 이 명령어들 중 하나를 실행하세요:

```
jekyll --version
gem list jekyll
```

RubyGems 를 사용해서 [Jekyll 의 최신 버전](https://rubygems.org/gems/jekyll)을 확인할 수 있습니다. 다른 방법으로는 `gem outdated` 명령을 실행해볼 수도 있습니다. 이 명령은 당신의 시스템에 설치된 루비 젬들 중 업데이트가 준비물의 목록을 보여줍니다. 가지고 있는 버전이 최신 버전이 아니라면, 이 명령을 실행하세요:

```
bundle update jekyll
```

Bundler 가 설치되어있지 않은 경우에는, 대신 다음과 같이 실행하세요:

```
gem update jekyll
```

Rubygems 를 최신 버전으로 업그레이드 하려면, 이렇게 실행하세요:

```
gem update --system
```

Jekyll 2.x 나 1.x 에서부터 업그레이드하는 경우에는 [업그레이드 페이지](https://jekyllrb-ko.github.io/docs/upgrading/)를 참고하세요.

<br>

<br>

## 프리릴리스 버전

프리릴리스 버전을 설치하려면, 모든 준비물이 올바르게 설치되었는지 확인한 후 다음 명령을 실행합니다:

```
gem install jekyll --pre
```

이 명령으로 가장 최신의 프리릴리스 버전이 설치됩니다. 만약 프리릴리스 중에서도 특정 버전이 필요하다면 `-v` 스위치를 사용하여 설치하려는 버전을 지정하세요:

```
gem install jekyll -v '2.0.0.alpha.1'
```

만약 개발 버전의 Jekyll 을 설치해보고 싶다면, 몇 가지 절차가 좀 더 필요합니다. 이를 통해 가장 뛰어난 최신의 기능들을 사용할 수 있게 됩니다만, 불안정할 수도 있습니다.

```
git clone git://github.com/jekyll/jekyll.git
cd jekyll
script/bootstrap
bundle exec rake build
ls pkg/*.gem | head -n 1 | xargs gem install -l
```

이제 필요한 모든 것들이 최신 버전으로 설치되었군요. 시작해봅시다!