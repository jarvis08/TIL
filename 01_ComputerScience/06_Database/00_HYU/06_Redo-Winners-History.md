# Redo-Winners/History

아래 내용은 두 가지 Recovery 과정 대한 설명이다. 두 가지 이론에서 사용되는 공통적인 요소는 winner/loser xact로, winner는 crash 이전에 commit이 완료된 xact, loser는 abort 되거나 fail된 xact를 의미한다.

<br><br>

## Redo-Winners

이 방법은 crash 이후 winner들에 대해서만 Redo를 시행하고, loser들에 대해서만 Undo를 시행하는 방법이다.

하지만 Redo-Winners 방법의 경우 중간에 abort된 xact에 대해 다시 한번 Undo를 시행해 버리고, 결국 abort 되지 않은 xact까지 abort 한 값으로 되돌린다.

<br><br>

## Redo-History

Redo-History는 ARIES(Algorithms for Recovery and Isolation Exploiting Semantics)로 처음 제안되었으며, 모든 xact들에 대해 Redo를 실행하고, abort 된 xact들(loser)에 대해 CLR(Compensation Log Record)을 적용하는 방법이다.

가장 먼저 physical하게 Redo를 시행하고, logical한 Undo를 시행한다.

이 방법은 abort한 xact에 대해 정확한 Undo가 가능하지만, crash가 계속될 경우 log의 길이가 계속해서 길어진다. 이전 crash에서 발행했던 CLR에 대해서도 CLR을 발행하고, 다시 CLR로 되돌리는 과정이 포함되기 때문이다. Log의 길이가 길어질 경우 recovery 시 탐색 시간이 길어져 비효율적이다.

따라서 **Next Undo**를 사용하여 CLR 사용시 다음으로 Undo할 대상을 기입하도록 한다. 이렇게 할 경우, crash가 반복되더라도 이미 발행한 CLR에 대해 CLR을 재발급할 필요가 없으며, 바로 다음 write 작업에 대해 Undo를 시행하면 된다.

