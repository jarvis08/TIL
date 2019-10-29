#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
using namespace std;

int T, ans, N, M;
int **B;
bool *visited;

class node {
public:
    int data;
    node *prev;
    node *next;
    node(int data) {
        this->data = data;
        this->prev = NULL;
        this->next = NULL;
    }
};
class stack {
private:
    int size;
    node *head;
    node *tail;
public:
    stack() {
        this->size = 0;
        this->head = NULL;
        this->tail = NULL;
    }
    void push(int data) {
        node *element = new node(data);
        if (size == 0) {
            this->head = element;
            this->tail = element;
            this->size++;
        }
        else {
            element->prev = this->tail;
            this->tail->next = element;
            this->tail = element;
            this->size++;
        }
    }

    int SIZE() {
        return this->size;
    }

    int pop() {
        int result;
        if (this->size == 1) {
            result = this->head->data;
            delete this->head;
            this->head = NULL;
            this->tail = NULL;
        }
        else {
            result = this->tail->data;
            this->tail = this->tail->prev;
            delete this->tail->next;
        }
        this->size--;
        return result;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie();
    freopen("input.txt", "r", stdin);
    cin >> T;
    for (int t = 1; t <= T; t++) {
        ans = 0;
        cin >> N;
        cin >> M;
        B = new int*[N];
        for (int i = 0; i < N; i++) {
            B[i] = new int[N];
            for (int j = 0; j < N; j++) {
                B[i][j] = 0;
            }
        }
        for (int i = 0; i < M; i++) {
            int a, b;
            cin >> a;
            cin >> b;
            B[a - 1][b - 1] = 1;
            B[b - 1][a - 1] = 1;
        }
        visited = new bool[N];
        for (int i = 0; i < N; i++) {
            visited[i] = 0;
        }
        for (int i = 0; i < N; i++) {
            if (visited[i] == false) {
                ans++;
                stack st = stack();
                st.push(i);
                visited[i] = true;
                while (st.SIZE() != 0) {
                    int t = st.pop();
                    for (int j = 0; j < N; j++) {
                        if (visited[j] == false && B[t][j] == 1) {
                            visited[j] = true;
                            st.push(j);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < N; i++) {
            delete B[i];
        }
    delete B;
    delete visited;
    cout << "#" << t << " " << ans << "\n";
    }
}