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

}

