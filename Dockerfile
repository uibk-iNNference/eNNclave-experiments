FROM alxshine/ennclave as ennclave-experiments

# install conda (taken from conda/miniconda3 Dockerfile)
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
 && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -bfp /usr/local \
 && rm -rf /tmp/miniconda.sh \
 && conda install -y python=3 \
 && conda update conda \
 && apt-get -qq -y remove curl bzip2 \
 && apt-get -qq -y autoremove \
 && apt-get autoclean \
 && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
 && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

WORKDIR /ennclave-experiments
COPY environment.yml /ennclave-experiments/
RUN conda env create

COPY docker_init.sh docker_ennclave_setup.sh docker_test.sh /ennclave-experiments/
COPY mnist /ennclave-experiments/mnist
COPY amazon /ennclave-experiments/amazon
COPY mit /ennclave-experiments/mit
COPY flowers /ennclave-experiments/flowers
COPY build_enclave.py experiment_utils.py time_enclave.py /ennclave-experiments/

CMD "bash"
# CMD ["bash", "tests.sh"]