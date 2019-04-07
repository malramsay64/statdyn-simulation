FROM malramsay/hoomd-core:2.3.4

# Which is required for pipenv to find the system python version
# TODO Work out how to remove this requirement
RUN yum install -y which && \
    yum clean all

# Install pip and pipenv
RUN python36 -m ensurepip && \
    python36 -m pip install -U pip

# Set environment encoding to UTF-8
# This is because the default encoding is typically ASCII which is a bad idea.
ENV LC_ALL=en_AU.utf8
ENV LANG=en_AU.utf8

ARG PYTHON=/usr/bin/python36

# No caching of files setting this to a falsy value.
# This helps keep the image smaller
ENV PIP_NO_CACHE_DIR=false

ADD ./ /srv/sdrun/

WORKDIR /srv/sdrun

RUN python36 -m pip install -e .[dev]
