SHELL=/bin/zsh

IMAGE_NAME=ex9
CONTAINER_NAME=ex9
MOUNT_PATH=/home/kajiwara21/work/B4Lecture-2021/ex_9/kajiwara
DATA_PATH=/home/kajiwara21/work/B4Lecture-2021/ex_9/dataset

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it \
		--shm-size=4g \
		--mount type=bind,source=$(MOUNT_PATH),target=/work \
		--mount type=bind,source=$(DATA_PATH),target=/work/dataset \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p 6007:6007 \
		$(IMAGE_NAME) /bin/bash

run-background:
	docker run -itd \
		--shm-size=4g \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p 6007:6007 \
		$(IMAGE_NAME) /bin/bash
	
start:
	docker start -it \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) /bin/bash

start-background:
	docker start -itd \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) /bin/bash

stop:
	docker stop $(CONTAINER_NAME)

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash

logs:
	docker logs $(CONTAINER_NAME)
