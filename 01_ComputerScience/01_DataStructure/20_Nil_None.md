# None, Nil, 그리고 친구들

출처 : <https://hamait.tistory.com/652>

Scalar 언어에서 사용되는 개념들이나, 유사하지 않을까 하여..

---

- `None`

  아무것도 없다는 리턴 값을 표현하기 위해 사용

  null 포인트 예외를 회피하기 위해 Option[T] 의 자식클래스로 사용 

- `Nil`

  아무것도 없는 List

- `Null`

  Trait

  모든 참조 타입(AnyRef를 상속한 모든 클래스) 의 서브클래스

  값 타입과는 호환성이 없다. 

- `null`

  Null 의 인스턴스이고 JAVA의 null 가 비슷하며, Python에서는 None으로 표현

- `Nothing`

  Trait이며, 모든것들의 서브타입

  기술적으로 예외는 Nothing 타입을 갖는다.

  이 타입은 값이 존재하지 않는다.

  값에 대입이 가능

  즉  리턴타입이 `Int` 인 메소드에서 리턴을 `Nothing` 타입인 예외를 던질 수 있다. 

- `Unit`

  아무 값도 리턴 하지 않는 메소드의 리턴타입으로 사용