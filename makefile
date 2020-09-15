IMG_NAME=pfmy

COMMAND_RUN=docker run \
	  --name ${IMG_NAME} \
	  --rm \
 	  -it \
	  --network host \
	  -v `pwd`/model:/home/developer/fmu

build:
	docker build --network host  --no-cache --rm -t ${IMG_NAME} .

remove-image:
	docker rmi ${IMG_NAME}

run:
	$(COMMAND_RUN) --detach=false ${IMG_NAME} /bin/bash -c "cd /home/developer/fmu && bash"