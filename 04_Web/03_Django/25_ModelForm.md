# ModelForm

Form Class 보다 몇 단계 더 편리한 **모듈화 기능**이며, Form Class를 완전히 대체한다.  **자동으로 `models.py`에 정의한 형태에 맞게 데이터를 생성하고 관리**해 주는 기능이다.

```python
# forms.py
from .models import Article
from django import forms

class ArticleModelForm(forms.ModelForm):
    class Meta:
        model = Article
        # '__all__' 모델의 모든 필드들을 가져오겠다는 의미
        # fields = '__all__'
        # exclude = ('title',) 타이틀 속성은 제외
        fields = ('title', 'content',)
        
        # widgets = {
        #     'title': forms.TextInput(
        #         label='제목'
        #         attrs={
        #             'class': 'form-control my-title',
        #             'placeholderform': '제목을 입력해주세요.',
        #             'id': 'title',
        #         }
        #     )
        # }
    
    # Meta class 안에서는 Field별 정의가 복잡(위의 widgets처럼 사용X)하므로,
    # 바깥에서 Field Customizing
    title = forms.CharField(
        max_length=20,
        label='제목',
        help_text='20자 이내로 입력해주세요.',
        widget=forms.TextInput(
            attrs={
                'class': 'form-control my-title',
                'placeholder': '제목을 입력해주세요.',
            }
        )
    )
    content = forms.CharField(
        label='내용',
        widget=forms.Textarea(
            attrs={
                'class': 'form-control my-content',
                'placeholder': '내용을 입력해주세요.',
                'rows': 5,
            }
        )
    )
```

```python
# views.py
from .forms import ArticleModelForm


def create(request):
    if request.method == 'POST':
        ####################################
        form = ArticleModelForm(request.POST)
        if form.is_valid():
            article = form.save()
            return redirect(article)
        ####################################
        else:
            return redirect('articles:create')
    else:
        form = ArticleModelForm()
        context = {
            'form': form,
        }
        return render(request, 'articles/create.html', context)
    
    
def update(request, article_pk):
    article = get_object_or_404(Article, pk=article_pk)
    if request.method == 'POST':
	##########################################################
        # instance를 지정해주지 않고 save를 하면, 수정이 아니라 새로 생성하게 된다.
        form = ArticleModelForm(request.POST, instance=article)
        if form.is_valid():
            form.save()
            return redirect(article)
    form = ArticleModelForm(instance=article)
    ##########################################################
    context = {
        'article': article,
        'form': form,
    }
    return render(request, 'articles/update.html', context)
```

