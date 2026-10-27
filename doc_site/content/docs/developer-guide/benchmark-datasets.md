---
title: Benchmark Datasets
type: docs
weight: 1110
prev: docs/developer-guide
next: docs/developer-guide/test-coverage
---

## Storage Location

Benchmark datasets are stored as assets of a release of the PETsARD GitHub repository:
- **Repository**: `moda-gov-tw/PETsARD`
- **Release tag**: `benchmark-datasets-v1`
- **Download URL**: `https://github.com/moda-gov-tw/PETsARD/releases/download/benchmark-datasets-v1/<filename>`

The data files are not part of the git history, so they do not add to the size of the repository; a single asset can be up to 2 GB.

## Available Datasets

The system automatically reads the list of available datasets from the `petsard/loader/benchmark_datasets.yaml` configuration file. Its `base_url` is the prefix of the download URL, and each dataset records its name, file name (`filename`), access type (`access`, only `public` is supported) and SHA-256 hash (`sha256`). A downloaded file must match the registered hash, otherwise loading reports an error.

## Adding a Dataset

1. Compute the SHA-256 hash of the file, for example with `sha256sum <filename>` (Linux / macOS) or `Get-FileHash <filename> -Algorithm SHA256` in PowerShell.
2. Open the `benchmark-datasets-v1` release on GitHub for editing and attach the file. Do not overwrite an uploaded asset; upload updated data under a new file name so that earlier PETsARD versions can still download the original file.
3. Add an entry to `benchmark_datasets.yaml` with the name, the file name, `access: public` and the hash.
4. Describe the dataset in the table of available datasets in [Benchmark Protocol]({{< ref "/docs/petsard-yaml/loader-yaml/benchmark-protocol" >}}).
