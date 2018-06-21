FROM ceshine/cuda-pytorch:0.4.0

MAINTAINER CeShine Lee <ceshine@ceshine.net>

RUN conda install -y jupyter && \
    conda clean -tipsy

RUN pip install --upgrade pip && \
    pip install jupyter_contrib_nbextensions tqdm docopt sk-video && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable collapsible_headings/main && \
    rm -rf ~/.cache/pip
