# [PYTORCH] Deep Q-learning for playing Procon



* Input File: Các thông tin ghi trong file input_xx.txt lần lượt:
	- Chiều cao và chiều rộng của boardgame: MxN
	- Ma trận điểm số của bảng đấu:
	- Số lượt chơi
	- Sô lượng agent mỗi đội: N_ag
	- Tọa độ lần lượt các agent của đội A và B (2*N_ag dòng) <x, y>
	- Số lượng kho báu: N_tr
	- Tọa độ lần lượt các kho báu (N_tr dòng) + phần thưởng <x, y, value>
	- Số lượng kho báu: N_wall
	- Tọa độ lần lượt các vật cản (N_wall dòng) <x, y>

## Run

To install the dependencies: 
```
pip install -r requirements.txt
```
Các môi trường làm việc:
python3
pytorch
pygame

With my code, you can:
* **Train your model from scratch** by running **python train.py**
