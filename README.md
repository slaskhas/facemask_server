
**mask detection using KERAS model in go**

# Work In Progress  
## not ready at all

Steps

1. Make sure to pull submodules
   git submodule update --init --recursive

2. Build a docker container that can be used to train the model
   and build the app.
   
   docker build -f Dockerfile.tensorgo -t tensorgo .

3. Run the container , mounting current directory

   docker run -it  -v "$PWD":/usr/src/myapp  tensorgo  /bin/bash

4. Add your own images to my_data/with_mask/ and  my_data/without_mask/
   This will improve the quality

5. Train the model on supplied data and your own data
   The model is a slightly modified MobileNet, with a few layers locked.
   
4. Convert the model into a "saved model" , suitable for golang.



build -f Dockerfile.tensorgo -t tensorgo .

docker run -it  -v "$PWD":/usr/src/myapp  tensorgo  /bin/bash




