# Start with ROS 2 Humble full desktop image
FROM osrf/ros:humble-desktop-full

# Args for User
ARG UNAME=user
ARG UID=1000
ARG GID=1000

# Ensure that installs are non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install setup utils and basic dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        iputils-ping \
        udev \
        usbutils \
        net-tools \
        wget \
        iproute2 \
        curl \
        nano \
        git \
        python3-pip \
        ros-humble-rqt* \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-tf-transformations \
        ros-humble-plotjuggler \
        ros-humble-plotjuggler-ros \
        ros-humble-pcl-ros \
        ros-humble-tf2-eigen \
        ros-humble-rviz2 \
        build-essential \
        libeigen3-dev \
        libjsoncpp-dev \
        libspdlog-dev \
        libcurl4-openssl-dev \
        cmake \
        python3-colcon-common-extensions \
     && apt purge -y --auto-remove \
     && rm -rf /var/lib/apt/lists/*
     
# Python3 Packages required by task allocation
RUN pip3 install \
    numpy \
    matplotlib \
    transforms3d \
    utm

# Create user
RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

# Allow the user to run sudo without a password
RUN echo "$UNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the non-root user for all other commands
USER $UNAME

# Install Dataspeed SDK (https://bitbucket.org/DataspeedInc/dbw_ros/src/ros2/) for Jeep
RUN /bin/bash -c "bash <(wget -q -O - https://bitbucket.org/DataspeedInc/dbw_ros/raw/ros2/dbw1/dbw_fca/scripts/sdk_install.bash)"

# Get Workspace Dependencies
RUN mkdir -p ~/MeasurementMCTS/src
COPY src /home/user/MeasurementMCTS/src
RUN cd ~/MeasurementMCTS && \
    sudo apt update &&\
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y
