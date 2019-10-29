#include "stdafx.h"
#include <iostream>
 
using namespace std;
 
struct User
{
    char name[20];
    char email[40];
};
 
template<typename T>
class Node
{
    template<typename T>
    friend class MyContainer;
private:
    Node* prev;
    Node* next;
    T* data;
public:
    Node() { this->prev = NULL; this->next = NULL; this->data = NULL; }
    Node(T* data) { this->prev = NULL; this->next = NULL; this->data = data; }
    Node(T* data, Node* prev, Node* next) { this->prev = prev; this->next = next; this->data = data; }
};
 
template<typename T>
class MyContainer
{
private:
    Node<T>* head = new Node<T>();
public:
    // 자신을 가리키도 데이터가 NULL인 헤더를 만든다.
    MyContainer() { this->head->prev = this->head; this->head->next = this->head; this->head->data = NULL; }
    ~MyContainer() {
        this->deleteContainer();
        delete this->head;
    }
 
    // 생성
    Node<T>* createNode(T* data)
    {
        // data를 저장할 동적메모리 할당
        T* newData = new T(*data);
 
        // node를 저장할 동적메모리 할당
        //  - newNode->prev = lastNode (prev는 마지막 노드를 가리킨다.)
        //  - newNode->next = firstNode (next는 첫번째 노드(head)를 가리킨다.)        
        Node<T>* newNode = new Node<T>(newData, this->head->prev, this->head);
         
        // 노드가 삽입될 양옆의 노드 정보를 변경한다.                
        if (this->head == this->head->next)
        {
            // 최초 생성일 경우
            this->head->next = newNode;
        }
        else
        {
            // 최초 생성이 아닐 경우
            this->head->prev->next = newNode;
        }
 
        // firstNode->prev (첫번째 노드의 prev는 새로운 노드를 가리킨다.)
        // lastNode->next (마지막 노드의 next는 새로운 노드를 가리킨다.)
        this->head->prev = newNode;
 
        return newNode;
    }    
 
    // 삭제
    void deleteNode(Node<T>* node)
    {
        // 이전노드와 다음노드를 연결한다.
        node->prev->next = node->next;
        node->next->prev = node->prev;
 
        // 자신을 메모리에서 삭제
        delete node->data;
        delete node;
    }
 
    // 컨테이너 삭제
    void deleteContainer()
    {
        if (this->head->next != this->head)
        {
            // head 만 남지 않았으면 다음노드 삭제하고 재귀호출
            this->deleteNode(this->head->next);
            this->printAll();
            this->deleteContainer();            
        }    
    }
 
    // 전체 출력 : 적용시 삭제
    void printAll()
    {    
        if (this->head != NULL)
        {
            cout << "\n### head Node ###\n";
            cout << "HEAD : " << this->head << "\n";
            cout << "PREV : " << this->head->prev << "\n";
            cout << "NEXT : " << this->head->next << "\n";
            cout << "DATA : " << this->head->data << "\n";
            cout << "\n";
 
            cout << "### Node List ###\n";
            if (this->head->next != NULL)
            {
                for (Node<T>* currentNode = this->head->next; currentNode != this->head; currentNode = currentNode->next)
                {
                    User* tmpUser = (User*)currentNode->data;
                    cout << "[<-" << currentNode->prev << "]";
                    cout << "[" << currentNode << "]";
                    cout << "[" << currentNode->next << "->]";
                    cout << " NAME : " << tmpUser->name << "\n";
                }
            }
        }
    }
};
 
int _tmain(int argc, _TCHAR* argv[])
{
    const int INSERT_SIZE = 5;
 
    MyContainer<User> myContainer;
    Node<User>* newNode[INSERT_SIZE];
     
    // Sample1. CreateNode : INSERT_SIZE 만큼 데이터를 입력
    for (int i = 0; i < INSERT_SIZE; i++)
    {
        User user;        
        cout << "insert name : ";
        cin.getline(user.name, 20);
        cout << "insert email : ";
        cin.getline(user.email, 40);
         
        newNode[i] = myContainer.createNode(&user);
    }
    myContainer.printAll();
 
    // Sample2. deleteNode : 2번째 노드를 삭제
    myContainer.deleteNode(newNode[1]);
    myContainer.printAll();
 
    // Sample3. deleteContainer : 전체 노드를 삭제 / 삭제해주지 않아도 소멸시 자동삭제되도록 구현한다.
    myContainer.deleteContainer();
 
    return 0;
}