# syntax=docker/dockerfile:1
#
# Build context for the image used by gerrytools' `mgrp` and `ben` modules
# (published as mgggdev/replicate). It bundles four engines behind a single
# image so end users never have to install R, Julia, and a Rust toolchain:
#
#   * rustrecom   - Rust ReCom            (gerrytools.mgrp.runners.recom;
#                   the deprecated frcw/frcw_* names are installed as shims)
#   * redist SMC  - R / C++ SMC           (gerrytools.mgrp.runners.smc)
#   * MSMS        - Julia Forest ReCom    (gerrytools.mgrp.runners.forest)
#   * ben         - BEN/XBEN codec + the smc_parser / msms_parser converters
#
# Build and publish via the Taskfile at the repo root (`task docker-build`,
# `task docker-push`). The rationale for each pinned version is noted inline
# next to its ARG below.
#
# Every engine version is exposed as a build ARG so it can be bumped in
# isolation, e.g.:
#   docker build --build-arg RUSTRECOM_REF=0.1.4 -t mgggdev/replicate:dev .

# Pin the multi-platform index so BuildKit selects the matching amd64 or arm64 image.
ARG R_BASE_DIGEST=sha256:555a0e7734b17f3901f01c8e379f87d797a0e6344a4cc3b246329ed3f0689809
ARG R_BASE_IMAGE=rocker/r-ver:4.6.1@${R_BASE_DIGEST}
FROM ${R_BASE_IMAGE}

ENTRYPOINT ["/bin/bash"]

# --- Engine versions ---------------------------------------------------------
# Engine sources are pinned to immutable commits so repeated builds use the same code.
ARG TARGETARCH
ARG UBUNTU_SNAPSHOT=20260722T120000Z
ARG CRAN_SNAPSHOT=2026-07-01
ARG RUST_VERSION=1.90.0
ARG RUSTUP_INIT_SHA256_AMD64=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10
ARG RUSTUP_INIT_SHA256_ARM64=9732d6c5e2a098d3521fca8145d826ae0aaa067ef2385ead08e6feac88fa5792
ARG JULIA_VERSION=1.10.2
ARG JULIA_SHA256_AMD64=51bccc9bb245197f24e6b2394e6aa69c0dc1e41b4e300b796e17da34ef64db1e
ARG JULIA_SHA256_ARM64=f319ff2812bece0918cb9ea6e0df54cc9412fc5ef8c0589b6a4fea485c07535d
ARG REDIST_VERSION=4.3.2                 # current CRAN release

ARG RUSTRECOM_REPO=https://github.com/mggg/rustrecom.git
# 0.1.4 branch: unified `rustrecom` CLI (chain/short-bursts/tilted subcommands,
# --config JSON mode on every subcommand, frcw compatibility shims, mst variant
# spellings with rmst deprecation shims).
ARG RUSTRECOM_REF=846971ca3205ce4286b42415d9627b5fd318b143

ARG MSMS_REPO=https://github.com/mggg/Multi-Scale-Map-Sampler.git
# Current main: path-based Writer, Float64 Measure, single-level support.
ARG MSMS_REF=0d77a358c7c7e714b14c52da13db49354c5b627a

# Parser refs take --config (inline JSON or a path), validate only the config
# fields they consume, and (msms) accept any number of hierarchy levels.
ARG SMC_PARSER_REPO=https://github.com/peterrrock2/smc_parser.git
ARG SMC_PARSER_REF=bda6f24989d79ab05e606c7453885a1b04e2d8af

ARG MSMS_PARSER_REPO=https://github.com/peterrrock2/msms_parser.git
ARG MSMS_PARSER_REF=e9353bec2c157aaf57d111064cb81918b7d9b1c1

# --- System dependencies -----------------------------------------------------
RUN sed -i \
    "s|http://archive.ubuntu.com/ubuntu/|\
https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/|g; \
     s|http://security.ubuntu.com/ubuntu/|\
https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/|g; \
     s|http://ports.ubuntu.com/ubuntu-ports/|\
https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/|g" \
    /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y \
    time \
    git \
    wget \
    build-essential \
    libssl-dev \
    pkg-config \
    curl \
    vim \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Rust toolchain ----------------------------------------------------------
RUN case "${TARGETARCH}" in \
        amd64) RUST_ARCH=x86_64-unknown-linux-gnu; \
            RUSTUP_INIT_SHA256="${RUSTUP_INIT_SHA256_AMD64}" ;; \
        arm64) RUST_ARCH=aarch64-unknown-linux-gnu; \
            RUSTUP_INIT_SHA256="${RUSTUP_INIT_SHA256_ARM64}" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL \
    "https://static.rust-lang.org/rustup/dist/${RUST_ARCH}/rustup-init" \
    -o /tmp/rustup-init \
    && echo "${RUSTUP_INIT_SHA256}  /tmp/rustup-init" | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rm /tmp/rustup-init
ENV PATH="/root/.cargo/bin:${PATH}"

# --- Julia -------------------------------------------------------------------
RUN case "${TARGETARCH}" in \
        amd64) JULIA_PATH=x64; JULIA_ARCH=x86_64; \
            JULIA_SHA256="${JULIA_SHA256_AMD64}" ;; \
        arm64) JULIA_PATH=aarch64; JULIA_ARCH=aarch64; \
            JULIA_SHA256="${JULIA_SHA256_ARM64}" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && JULIA_MINOR="$(echo ${JULIA_VERSION} | cut -d. -f1-2)" \
    && ARCHIVE="julia-${JULIA_VERSION}-linux-${JULIA_ARCH}.tar.gz" \
    && wget \
    "https://julialang-s3.julialang.org/bin/linux/${JULIA_PATH}/${JULIA_MINOR}/${ARCHIVE}" \
    && echo "${JULIA_SHA256}  ${ARCHIVE}" | sha256sum -c - \
    && tar zxvf "${ARCHIVE}" \
    && rm "${ARCHIVE}" \
    && mv "julia-${JULIA_VERSION}" /opt \
    && ln -s "/opt/julia-${JULIA_VERSION}/bin/julia" /usr/local/bin/julia

# --- R / redist (SMC) --------------------------------------------------------
# R's INSTALL helper lacks a shebang. Native shells retry it through /bin/sh,
# but BuildKit's QEMU wrapper returns an exec-format error without that fallback.
RUN sed -i '1i#!/bin/sh' /usr/local/lib/R/bin/INSTALL \
    && head -n 1 /usr/local/lib/R/bin/INSTALL | grep -qx '#!/bin/sh'

# Split from the redist install so dependency rebuilds stay cached. Both layers use a dated
# P3M snapshot, and dependencies=NA excludes test and documentation dependencies.
ENV CRAN="https://p3m.dev/cran/__linux__/noble/${CRAN_SNAPSHOT}"
RUN R -e \
    "install.packages(c('argparser', 'dplyr', 'ggplot2', 'jsonlite', 'sf'), \
    dependencies=NA, repos=Sys.getenv('CRAN'))"
RUN R -e \
    "install.packages('redist', dependencies=NA, repos=Sys.getenv('CRAN')); \
    stopifnot(packageVersion('redist') == package_version('${REDIST_VERSION}'))"

# --- Format parsers ----------------------------------------------------------
# These convert SMC / MSMS output into JSONL or BEN and are required by the smc
# and forest runners. They vendor their own binary-ensemble dependency to emit
# .ben; the binary-ensemble 1.0.0 Python package gerrytools.ben uses auto-detects
# the stream variant, so this output stays readable. The cleaner path is to make
# the parsers emit JSONL and encode in Python, which removes the format coupling
# and these tags' relevance.
#
# The standalone `ben` CLI is intentionally NOT installed: gerrytools.ben now
# uses the binary-ensemble Python bindings instead of this image, so nothing in
# gerrytools shells out to `ben` anymore. To restore it for debugging:
#   RUN cargo install binary-ensemble --version <X>
RUN cargo install --locked --git ${MSMS_PARSER_REPO} --rev ${MSMS_PARSER_REF}
RUN cargo install --locked --git ${SMC_PARSER_REPO} --rev ${SMC_PARSER_REF}

# --- rustrecom (Rust ReCom) -------------------------------------------------------
RUN git clone ${RUSTRECOM_REPO} /home/rustrecom \
    && cd /home/rustrecom \
    && git checkout ${RUSTRECOM_REF} \
    && cargo install --locked --path .

# --- Forest ReCom (Julia MSMS) -----------------------------------------------
RUN git clone ${MSMS_REPO} /home/forest \
    && cd /home/forest \
    && git checkout ${MSMS_REF}

# Overlay the mgrp-maintained CLI glue on top of the cloned engine trees. These
# scripts live in this repo (docker/cli/) so they are versioned alongside the
# Python runners that invoke them, rather than inside the upstream engines.
COPY cli/forest/multi_cli.jl   /home/forest/cli/multi_cli.jl
COPY cli/forest/run_sampler.jl /home/forest/cli/run_sampler.jl
COPY cli/smc/smc_cli.R         /home/smc/cli/smc_cli.R

# Instantiate MSMS and add the CLI dependencies to its project.
RUN julia --project=/home/forest -e \
    'using Pkg; Pkg.instantiate(); Pkg.add(name="ArgParse", version="1.2.0"); \
    Pkg.add(name="JSON", version="0.21.4"); Pkg.build()'
