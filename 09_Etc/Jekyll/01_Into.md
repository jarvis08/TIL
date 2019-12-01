# 기본 사용법

루비 젬 Jekyll 은 터미널 창에서 사용할 수 있는 실행파일 `jekyll` 을 만들어줍니다. 이 명령은 다양한 방식으로 사용할 수 있습니다:

```
jekyll build
# => 현재 폴더의 컨텐츠를 가지고 ./_site 에 사이트를 생성합니다.

jekyll build --destination <destination>
# => 현재 폴더의 컨텐츠를 가지고 <destination> 에 사이트를 생성합니다.

jekyll build --source <source> --destination <destination>
# => <source> 폴더의 컨텐츠를 가지고 <destination> 에 사이트를 생성합니다.

jekyll build --watch
# => 현재 폴더의 컨텐츠를 가지고 ./_site 에 사이트를 생성합니다.
#    변경사항이 감지되면, 자동으로 다시 생성합니다.
```

<br>

<br>

## 개발 환경설정 기본값 변경하기

개발환경의 URL 기본값은 `http://localhost:4000` 으로 설정되어 있습니다. 3.3.0

운영환경에 맞춰 생성하고자 하는 경우에는:

- `_config.yml` 에 운영환경 URL 을 설정합니다. 예시, `url: https://example.com`.
- `JEKYLL_ENV=production bundle exec jekyll build` 를 실행합니다.

<br>

### 자동으로 다시 생성할 때, `_config.yml` 의 변경사항은 반영되지 않습니다.

주 환경설정 파일인 `_config.yml` 에는 전역 환경설정과 변수들이 정의되어 있으며, 실행 시점에 한 번만 읽어들입니다. 자동 재생성을 사용하는 중이라도, 완전히 새로 실행하기 전까지는 `_config.yml` 의 변경사항을 읽어들이지 않습니다.

자동 재생성 과정에서 [데이터 파일](https://jekyllrb-ko.github.io/docs/datafiles)은 다시 읽어들입니다.

<br>

### Site Destination 폴더는 사이트 빌드 시 초기화됩니다

사이트 빌드 시에 자동으로 `` 안의 파일들을 지우는 것이 디폴트로 설정되어 있습니다. 사이트에서 생성하지 않는 파일들은 모두 사라질 것입니다. 환경설정 옵션 `` 를 사용해 `` 에 그대로 옮길 파일이나 폴더를 지정할 수 있습니다.

중요한 디렉토리는 절대 `` 으로 지정하면 안됩니다; 웹 서버로 옮기기 전에 임시로 파일들을 보관할 경로를 입력하세요.

Jekyll 은 개발 서버도 내장하고 있어서, 로컬상에서 브라우저로 접속하여 사이트가 어떻게 생성될지 미리 살펴볼 수 있습니다.

```
jekyll serve
# => 개발서버가 실행됩니다. http://localhost:4000/
# 자동 재생성: 활성화. 비활성화하려면 `--no-watch` 를 사용하세요.

jekyll serve --livereload
# 변경사항이 발생했을 때 LiveReload 기능이 브라우저를 새로고침합니다.

jekyll serve --incremental
# 재생성 소요시간을 줄이기 위해 증분 재생성 기능으로 부분 빌드를 합니다.

jekyll serve --detach
# => `jekyll serve` 와 동일하지만 현재 터미널에 독립적으로 실행됩니다.
#    서버를 종료하려면, `kill -9 1234` 를 실행하세요. "1234" 는 PID 입니다.
#    PID 를 모르겠다면, `ps aux | grep jekyll` 를 실행하고 해당 인스턴스를 종료하세요.
jekyll serve --no-watch
# => `jekyll serve` 와 동일하지만 변경사항을 감시하지 않습니다.
```

이것들은 [환경설정 옵션](https://jekyllrb-ko.github.io/docs/configuration/)의 극히 일부분만 보여준 것입니다. 다양한 환경설정 옵션들을 위와 같이 명령행에 플래그로 지정할 수 있고 또 다른 (그리고 더 일반적인) 방법으로, 루트 디렉토리의 `_config.yml` 파일에 지정할 수 있습니다. Jekyll 이 작동하기 시작하면, 자동적으로 이 파일에 지정한 옵션들을 사용합니다. 예를 들어, `_config.yml` 파일에 다음과 같이 입력한다면:

```
source:      _source
destination: _deploy
```

아래 두 명령은 동일합니다:

```
jekyll build
jekyll build --source _source --destination _deploy
```

환경설정 옵션에 대한 더 자세한 내용은 [환경설정](https://jekyllrb-ko.github.io/docs/configuration/) 페이지를 참고하세요.

<br>

### 도움을 요청하세요

사용할 수 있는 모든 옵션과 사용법을 알려주는 `help` 명령은 언제든지 사용할 수 있으며, 이 명령은 하위 명령어인 `build`, `serve`, `new` 와 함께 사용할 수도 있습니다. 예시, `jekyll help new` 또는 `jekyll help build`.

만약 이 문서처럼 끊임없이 수정되는 최신 문서에 관심이 많다면, `jekyll-docs` gem 을 설치하고, 터미널에서 `jekyll docs` 를 실행하세요.

<br>

<br>

## 디렉토리 구조

Jekyll 의 핵심 역할은 텍스트 변환 엔진입니다. 시스템의 컨셉은 다음과 같습니다: 당신이 마크다운이나 Textile 또는 일반 HTML 등 자신이 즐겨 사용하는 마크업 언어로 문서를 작성하면, Jekyll 은 이 문서들을 하나 또는 여러 겹의 레이아웃으로 포장합니다. 사이트 URL 구성 방식이나 어떤 데이터를 레이아웃에 표시할 것인지 등, 변환 과정에 포함된 다양한 동작들은 당신이 원하는대로 조정할 수 있습니다. 단지 텍스트 파일을 수정하는 것만으로 이 모든 일들이 가능합니다. 그 결과로 정적 웹 사이트가 만들어집니다.

가장 기본적인 Jekyll 사이트는 보통 이렇게 생겼습니다:

```
.
├── _config.yml
├── _data
|   └── members.yml
├── _drafts
|   ├── begin-with-the-crazy-ideas.md
|   └── on-simplicity-in-technology.md
├── _includes
|   ├── footer.html
|   └── header.html
├── _layouts
|   ├── default.html
|   └── post.html
├── _posts
|   ├── 2007-10-29-why-every-programmer-should-play-nethack.md
|   └── 2009-04-26-barcamp-boston-4-roundup.md
├── _sass
|   ├── _base.scss
|   └── _layout.scss
├── _site
├── .jekyll-metadata
└── index.html # 'index.md' 이어도 되지만 올바른 YAML 머리말이 필요합니다
```

<br>

### 루비 젬 기반 테마를 사용하는 Jekyll 사이트의 디렉토리 구조

**Jekyll 3.2** 버전부터, `jekyll new` 명령으로 생성된 Jekyll 프로젝트는 [루비 젬 기반 테마](https://jekyllrb-ko.github.io/docs/themes/)를 사용하여 사이트의 외관을 구성합니다. 이로 인해, 테마 루비 젬에 기본적으로 포함된 경량 디렉토리 구조 : `_layouts`, `_includes`, `_sass` 를 갖게 됩니다.

현재의 기본 테마는 [minima](https://github.com/jekyll/minima) 이며, `bundle show minima` 명령으로 Minima 테마의 파일들이 어디에 저장되어 있는지 볼 수 있습니다.

각 파일과 디렉토리가 하는 일의 개요는 다음과 같습니다:

| 파일 / 디렉토리                                              | 설명                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `_config.yml`                                                | [환경설정](https://jekyllrb-ko.github.io/docs/configuration/) 정보를 보관한다. 명령어를 실행할 때 여러가지 옵션들을 추가할 수도 있지만, 그렇게 따로 외우는 것보다 이 파일에 정의해두는게 더 편리하다. |
| `_drafts`                                                    | 초안이란 아직 게시하지 않은 포스트를 말한다. 파일명 형식에 날짜가 없다: `title.MARKUP`. 사용 방법은 [초안 활용하기](https://jekyllrb-ko.github.io/docs/drafts/)를 참고하라. |
| `_includes`                                                  | 재사용하기 위한 파일을 담는 디렉토리로서, 필요에 따라 포스트나 레이아웃에 쉽게 삽입할 수 있다. `{% include file.ext %}` 와 같이 Liquid 태그를 사용하면 `_includes/file.ext` 파일에 담긴 코드가 삽입된다. |
| `_layouts`                                                   | 포스트를 포장할 때 사용하는 템플릿이다. 각 포스트 별로 레이아웃을 선택하는 기준은 [YAML 머리말](https://jekyllrb-ko.github.io/docs/frontmatter/)이며, 자세한 내용은 다음 섹션에서 설명한다. `{{ content }}` 와 같이 Liquid 태그를 사용하면 페이지에 컨텐츠가 주입된다. |
| `_posts`                                                     | 한마디로 말하면, 당신의 컨텐츠다. 중요한 것은 파일들의 명명규칙인데, 반드시 이 형식을 따라야 한다: `YEAR-MONTH-DAY-title.MARKUP`. [고유주소](https://jekyllrb-ko.github.io/docs/permalinks/)는 포스트 별로 각각 정의할 수 있지만, 날짜와 마크업 언어 종류는 오로지 파일명에 의해 결정된다. |
| `_data`                                                      | 사이트에 사용할 데이터를 적절한 포맷으로 정리하여 보관하는 디렉토리. Jekyll 엔진은 이 디렉토리에 있는 (확장자와 포맷이 `.yml` 또는 `.yaml`, `.json`, `.csv` 인) 모든 데이터 파일을 자동으로 읽어들여 `site.data` 로 사용할 수 있도록 만든다. 만약 이 디렉토리에 `members.yml` 라는 파일이 있다면, `site.data.members` 라고 입력하여 그 컨텐츠를 사용할 수 있다. |
| `_sass`                                                      | Sass 조각파일들로, 프로젝트의 `main.scss` 에 임포트할 수 있으며 임포트 후에는 다시 하나의 스타일시트(`main.scss`)로 가공되어 사이트에 사용되는 스타일들을 정의한다. |
| `_site`                                                      | Jekyll 이 변환 작업을 마친 뒤 생성된 사이트가 저장되는 (디폴트) 경로이다. 대부분의 경우, 이 경로를 `.gitignore` 에 추가하는 것은 괜찮은 생각이다. |
| `.jekyll-metadata`                                           | Jekyll 은 이 파일을 참고하여, 마지막으로 빌드한 이후에 한번도 수정되지 않은 파일은 어떤 것인지, 다음 빌드 때 어떤 파일을 다시 생성해야 하는지 판단할 수 있다. 생성된 사이트에 이 파일이 복사되지는 않는다. 대부분의 경우, 이 파일을 `.gitignore` 에 추가하는 것은 괜찮은 생각이다. |
| `index.html` 또는 `index.md` 및 다른 HTML, 마크다운, Textile 파일 | Jekyll 은 [YAML 머리말](https://jekyllrb-ko.github.io/docs/frontmatter/) 섹션을 가진 모든 파일을 찾아 변환 작업을 수행한다. 위에서 언급하지 않은 다른 디렉토리나 사이트의 루트 디렉토리에 있는 모든 `.html`, `.markdown`, `.md`, `.textile` 이 여기에 해당한다. |
| 다른 파일/폴더                                               | `css` 나 `images` 폴더, `favicon.ico` 파일같이 앞서 언급하지 않은 다른 모든 디렉토리와 파일들은 있는 그대로 생성된 사이트에 복사한다. 다른 사람들이 만든 사이트는 어떤식으로 생겼는지 궁금하다면, [Jekyll 을 사용하는 사이트들](https://jekyllrb-ko.github.io/docs/sites/)이 이미 많이 있으니 살펴보도록 한다. |