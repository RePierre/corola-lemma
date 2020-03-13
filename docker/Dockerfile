FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    openssh-server \
    tmux \
    emacs \
    emacs-goodies.el \
    curl \
    git \
    python3 \
    python3-pip \
    python3-virtualenv \
    python3-dev \
    build-essential \
    locales \
    wget \
    unzip \
    htop

# From https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy scipy tensorflow scikit-learn keras gensim matplotlib pyyaml spacy pandas seaborn && \
    pip install elpy jedi rope yapf importmagic flake8 autopep8 black

RUN cd /root/ && \
    git clone https://github.com/RePierre/.emacs.d.git .emacs.d && \
    echo "(setq package-check-signature nil)" >> ./.emacs.d/config.el && \
    sed -i '/^.*emacs-init.*$/d' ./.emacs.d/init.el && \
    sed -i 's/(shell . t)/(sh . t)/' ./.emacs.d/common-config.org && \
    sed -i '/^.*org-config.*$/d' ./.emacs.d/init.el && \
    sed -i 's/\:defer\ t//' ./.emacs.d/python-config.org && \
    sed -i '/(setq auto-package-update-prompt-before-update t)/d' ./.emacs.d/emacs-init.org && \
    sed -i 's/(auto-package-update-maybe)//' ./.emacs.d/emacs-init.org && \
    echo "(setq elpy-rpc-virtualenv-path \"$VIRTUAL_ENV\") (pyvenv-activate \"$VIRTUAL_ENV\")" >> ./.emacs.d/after-init.el

RUN emacs --daemon

# Set terminal colors https://stackoverflow.com/a/64585/844006
ENV TERM=xterm-256color

# https://serverfault.com/a/689947
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    sed -i -e 's/# ro_RO.UTF-8 UTF-8/ro_RO.UTF-8 UTF-8/' /etc/locale.gen && \
    echo 'LANG="en_US.UTF-8"'>/etc/default/locale && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

# http://jaredmarkell.com/docker-and-locales/
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN mkdir -p /tmp /data/corpora /output

RUN cd /tmp && \
	wget http://89.38.230.23/word_embeddings/we/corola.300.20.vec.zip && \
	mv /tmp/corola.300.20.vec.zip /data/corola-word-embeddings.vec.zip && \
	wget http://89.38.230.23/word_embeddings_lemma/we/corola.300.50.vec.zip && \
	mv /tmp/corola.300.50.vec.zip /data/corola-lemma-embeddings.vec.zip

# RUN cd /tmp && \
# 	wget https://relate.racai.ro/marcell/ro-xml.zip && \
# 	mv /tmp/ro-xml.zip /data/corpora/marcell.zip

ADD word-lemmas.csv /data/

ARG password

RUN mkdir /var/run/sshd
RUN echo "root:${password}" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# From https://stackoverflow.com/a/42125241/844006
ARG ssh_prv_key
ARG ssh_pub_key
# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
