# Tips

어마어마하게 큰, 정렬되어 있지 않은 배열에서 k번째로 작은 값 구하기

- 이진 탐색의 경우 이미 정렬되어 있는 상태에서 사용할 수 있으므로 제외

- k selection의 경우 O(n^2)의 시간 복잡도를 가진다.

  가장 작은 값부터 찾아 나간다.

- k-quick selection을 사용하면 쉽게 찾을 수 있다.

  일반 퀵 정렬처럼 피봇 설정 후, 피봇 값의 제 위치를 찾는 과정을 거친다.

  피봇의 위치 인덱스를 찾은 후, k 가 피봇보다 크거나 작으므로, 나머지 부분을 버려서 탐색 범위를 좁히며 탐색

<br>

### Google Python Style Guide

![Google_Convention_NameSpace](/Users/whdbin/Documents/00_TIL/02_ComputerScience/assets/Google_Convention_NameSpace.png)