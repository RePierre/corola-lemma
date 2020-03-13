#!/bin/bash

sed -i '/^\[localhost\].*$/d' ~/.ssh/known_hosts

docker build -t corola-lemma:development \
       --build-arg ssh_prv_key="$(cat ~/.ssh/id_rsa)" \
       --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)" \
       --build-arg password=$1 \
       .
