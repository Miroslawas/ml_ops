FROM python:3.7-stretch

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.22.0 pandas flask gunicorn catboost==0.24.0 pandas_profiling pyprojroot

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY opt/program /opt/program

WORKDIR /opt/program