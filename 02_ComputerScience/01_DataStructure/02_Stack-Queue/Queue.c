#include <stdio.h>

#define QUEUE_SIZE 10

typedef struct Queue
{
    int que[QUEUE_SIZE];
    int head;
    int tail;
}Queue;

void Enqueue(Queue *queue, int element);
int Dequeue(Queue *queue);
int isFull(Queue *queue);
int isEmpty(Queue *queue);
void initQueue(Queue *queue);

int main(void)
{
    Queue queue;
    initQueue(&queue);
    int i;

    for(i=1;i<=8;i++)
        Enqueue(&queue, i);
    printf("head = %d\ntail = %d\n", queue.head, queue.tail);
    while(i > 5){
        Dequeue(&queue);
        i--;}
    printf("head = %d\ntail = %d\n", queue.head, queue.tail);
    while(i<12){
        Enqueue(&queue, i);
        i++;}
    printf("head = %d\ntail = %d\n", queue.head, queue.tail);
}

void Enqueue(Queue *queue, int element){
    printf("Try Enqueue %d\n", element);
    if(isFull(queue)){
        printf("Overflow\n");
        return ;}
    queue->que[queue->tail] = element;
    if(queue->tail == QUEUE_SIZE){
        queue->tail = 1;
        return ;}
    queue->tail ++;}

int Dequeue(Queue *queue){
    printf("Try Dequeue\n");
    int temp = 0;
    if(isEmpty(queue)){
        printf("Underflow\n");
        return temp;}
    temp = queue->que[queue->head];
    if(queue->head == QUEUE_SIZE){
        queue->head = 1;
        return temp;}
    queue->head ++;
    return temp;}

int isFull(Queue *queue){
    return (queue->head == queue->tail + 1) || (queue->head == 1 && queue->tail == QUEUE_SIZE);
}

int isEmpty(Queue *queue){
    return queue->head == queue->tail;
}

void initQueue(Queue *queue){
    printf("Initiate Queue\n");
    queue->head = 1;
    queue->tail = 1;
}