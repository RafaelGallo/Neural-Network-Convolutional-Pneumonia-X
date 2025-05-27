# Neural_Network_Convolutional documentation!

## Description

Pneumonia X-Ray Classification with CNNs

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://d1/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://d1/data/` to `data/`.


