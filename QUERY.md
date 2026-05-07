@acoustic_side_channel_attack.pdf 이 논문을 구현할 거야. 근데, 성능 개선을 위해서 @acoustic_side_channel_attack.pdf 논문에서 사용했던 CoAtNet을 @MaxViT.pdf MaxVit 논문에 나오는 MaxViT-S로 교체해서 구현할 거야.
         
@README.md 에 나온 Isolating Keystrokes 부분의 코드를 참고해서 @MBPWavs/ 에 있는 .wav 파일을 분리할 거야. 각 wav 파일은 제목에 해당하는 키에 대한 audio 25번이 연속으로 들어가 있어. 데이터 전처리 과정을 @acoustic_side_channel_attack.pdf 과정과 동일하게 거친 뒤, MaxViT-S 구조를 학습시키는 python 코드를 짜야해. 계획을 수립해줘. PLAN.md 파일을 만들어서 계획을 저장해.

필요한 library가 있다면 requirements.txt 파일을 만들어줘.


