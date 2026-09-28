Level 1: rag.py
Level 2: manual poisoning
Level 3: pretrained Confundo
Level 4: eval_level4.py

***
KHuyến khích ae chạy trong Linux/Ubuntu.

AE chạy môi trường ảo trước: source .venv/bin/activate

Rồi chạy từng Level một nhá. 

Ví dụ chạy level 2 sẽ là: python raglv2.py data/poisoned_docs.json

Trong Ubuntu trông sẽ như thế này:
(.venv) tony@DESKTOP-9UB03SL:~/mini-confundo$ python raglv2.py data/poisoned_docs.json
