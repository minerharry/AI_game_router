# syntax=docker/dockerfile:1
FROM rayproject/ray:nightly-py310
WORKDIR /router
COPY no_ray_requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/minerharry/AI_game_router.git --branch ray_parallelization