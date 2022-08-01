# FROM ubi8/python-39

# If the container is not run on a registered RHEL system then use this for the base container:
FROM registry.fedoraproject.org/f35/python3

LABEL maintainer="tbrunell@redhat.com"
USER 0

# If you have a GPU
# RUN yum -y install xorg-x11-drv-nvidia-cuda-libs.x86_64

RUN yum -y install --nodocs opencv-core opencv-contrib  &&\
    yum clean all -y

RUN python3 -m pip install --upgrade pip
WORKDIR /app
RUN mkdir /app/model
RUN mkdir /app/templates/
RUN pip install tensorflow \
                tensorflow \
                tensorflow-hub \
                opencv-python \
                matplotlib \
                flask

EXPOSE 8080

COPY app.py /app
COPY MoveNet.py /app
COPY templates/ /app/templates/
COPY model/ /app/model

CMD ["python3","app.py"]
