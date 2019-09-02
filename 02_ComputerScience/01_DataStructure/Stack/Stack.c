#include <stdio.h>

// Stack Size 정의
#define STACK_SIZE 10

// Stack Structure 구조체 정의
typedef struct Stack
{
    // 저장소 사이즈
    int buf[STACK_SIZE];
    // 가장 최근 요소를 지정하는 top
    int top;
}Stack;

// Stack에서 가장 위(top)에 요소를 삽입하는 Push
// Stack 구조체의 *stack에 element를 추가
void Push(Stack *stack, int element);
// Stack에서 가장 나중에 들어온 요소를 삭제하는 Pop
int Pop(Stack *stack);

// 배열로 스택을 구현 시, 저장소가 꽉 찼는지 확인하거나 비었는지 확인 가능
int isFull(Stack *stack);
int isEmpty(Stack *stack);

// top = -1로 Stack을 초기화
void initStack(Stack *stack);

int main(void)
{
    int i;
    // Stack 구조체를 통해 stack 생성
    Stack stack;
    // stack을 초기화
    initStack(&stack);

    for(i=1;i<=5;i++)
    {
        Push(&stack, i);
    }

    while(!isEmpty(&stack))
        printf("%d\n", Pop(&stack));
    printf("\n");

    // malloc을 이용하여 동적 메모리 할당시, free(객체명)을 이용하여 동적 메모리 해제
    // free(stack);
    return 0;
}

void Push(Stack *stack, int element)
{
    if(isFull(stack))
    {
        printf("Overflow\n");
        return ;
    }
    stack->buf[stack->top] = element;
    stack->top++;
    printf("Push element %d\n", element);
}

int Pop(Stack *stack)
{
    int element = 0;
    if(isEmpty(stack))
    {
        printf("Underflow\n");
        return element;
    }
    element = stack->buf[stack->top];
    printf("Pop element %d\n", element);
    stack->top--;
    return element;
}

int isFull(Stack *stack)
{
    return (stack->top+1) == STACK_SIZE;
}

int isEmpty(Stack *stack)
{
    return stack->top == -1;
}

void initStack(Stack *stack)
{
    stack->top = -1;
}