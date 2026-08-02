# Setting up Docker

The {mod}`gerrytools.mgrp` module runs native redistricting engines written in Rust, Julia, R,
and C++ inside a container. Docker supplies those engines and their system dependencies in a
known environment, while GerryTools prepares the command, mounts the input and output paths, and
returns the generated files to Python.

This arrangement saves you from installing each sampler toolchain separately. It also makes a run
more reproducible because GerryTools uses a versioned Docker image pinned to an immutable digest.

## What you need

A working MGRP installation has three pieces:

1. GerryTools with the optional `mgrp` dependencies.
2. Docker Desktop, Docker Engine, or another compatible daemon that is running.
3. Access to the versioned `mgggdev/replicate` image used by GerryTools.

Installing the Python package does not install or start Docker. Conversely, installing Docker does
not add the Docker SDK to the Python environment used by your script or notebook.

## Install the Python support

Install the `mgrp` extra in the same environment that will run the analysis:

```console
uv add "gerrytools[mgrp]"
```

For a pip-managed environment, use:

```console
python -m pip install "gerrytools[mgrp]"
```

The extra includes the Docker Python SDK. Using `python -m pip` avoids accidentally installing the
SDK for a different Python interpreter.

## Install and start Docker

Download Docker from the official [Get Docker](https://docs.docker.com/get-started/get-docker/)
page. Docker Desktop is the simplest choice on macOS and Windows. On Linux, either Docker Desktop
or Docker Engine works.

```{figure} ../_static/images/docker/docker_download.png
:alt: Docker's download page with the platform download control
:width: 100%

Choose the download for your operating system and processor. The website may look different from
this screenshot, but the platform selector serves the same purpose.
```

If this is your first time installing Docker, you will likely be greeted by the following screens:

```{figure} ../_static/images/docker/docker-subscription-agreement.png
:alt: Docker Desktop subscription agreement
:width: 80%

Screen 1: The Docker Desktop subscription agreement that appears when first installing Docker Desktop.
```


```{figure} ../_static/images/docker/docker-settings-warning.png
:alt: Docker Desktop settings warning
:width: 80%

Screen 2: The Docker Desktop settings warning that appears on initial setup fo some docker instances.
```


```{figure} ../_static/images/docker/docker-signin.png
:alt: Docker Desktop signin page
:width: 80%

Screen 3: The Docker Desktop signin page that appears after dismissing the subscription agreement.
```

You may close the subscription agreement and skip the sign-in. You should not need either of them 
to run GerryTools. The recommended settings should also be fine.

After installation, open Docker Desktop and wait for the engine to finish starting. The dashboard
shows containers, images, volumes, and the current engine status.

```{figure} ../_static/images/docker/docker-desktop-hero.png
:alt: Docker Desktop dashboard showing running containers
:width: 80%
```

A Docker account is not normally required to run GerryTools or pull its public image. Signing in
can provide higher pull limits, and an organization-managed Docker installation may require it.

```{important}
The Docker daemon must remain running for the entire MGRP run. On macOS and Windows, this usually
means leaving Docker Desktop open. If a previously working run suddenly cannot connect, confirm
the engine status or restart Docker Desktop before changing the runner configuration.
```

### macOS

Install Docker Desktop for the correct processor: Apple silicon or Intel. GerryTools supports
Docker Desktop 4.63.0 or later. Docker currently supports the latest major macOS release and the
two previous releases, so consult Docker's
[macOS installation requirements](https://docs.docker.com/desktop/setup/install/mac-install/)
when an installer rejects an older system.

MGRP exchanges graph, output, and log files through bind mounts. On Docker Desktop for Mac,
VirtioFS is the default file-sharing implementation and is usually the best option. In Docker
Desktop 4.63, the setting appeared under **Settings > General** as shown below. Newer releases may
place it under the virtual machine or file-sharing settings.

```{figure} ../_static/images/docker/docker-VirtioFS.png
:alt: Docker Desktop settings on macOS with VirtioFS selected
:width: 100%
```

If Docker reports that a mount is denied, open **Settings > Resources > File sharing** and confirm
that the directory containing the analysis is shared. Projects under `/Users` are normally covered
by the default settings.

### Windows

Docker Desktop's WSL 2 backend is the recommended setup for most Windows users. Before installing
Docker Desktop, open PowerShell and check WSL:

```powershell
wsl --version
wsl --update
```

Docker currently requires WSL 2.1.5 or later. The machine must also have hardware virtualization
enabled in its BIOS or UEFI settings. If Docker reports that virtualization is unavailable, follow
the computer manufacturer's instructions rather than changing unrelated firmware settings.

During installation, select the WSL 2 backend and run Linux containers. The MGRP image contains
Linux versions of the samplers. Docker's
[Windows installation guide](https://docs.docker.com/desktop/setup/install/windows-install/)
lists the supported Windows versions and the complete WSL and hardware requirements.

If the analysis runs inside a WSL distribution, enable Docker Desktop integration for that
distribution. Keep the project and its inputs in a location visible to the same environment that
runs Python. Crossing repeatedly between the Windows and Linux filesystems can make large mounted
datasets noticeably slower.

### Linux

Install Docker Engine from Docker's
[distribution-specific instructions](https://docs.docker.com/engine/install/) or install Docker
Desktop for Linux. Start the daemon before running GerryTools:

```console
sudo systemctl start docker
```

If `docker version` works only with `sudo`, the Python SDK will usually encounter the same socket
permission problem. Follow Docker's
[Linux post-installation guide](https://docs.docker.com/engine/install/linux-postinstall/) or use
Docker's rootless mode. Membership in the `docker` group grants root-level control of the host
through the daemon, so add users deliberately.

Do not run the entire analysis as root to work around a socket error. Files written into mounted
output directories can then be owned by root, creating a separate permissions problem.

## Verify the installation

Check each layer in order. This makes it clear whether a failure comes from the Docker client, the
daemon, or the Python environment.

### 1. Check the client and daemon

Run:

```console
docker version
docker info
```

`docker version` should report both **Client** and **Server** sections. A client-only response means
the command-line program is installed but cannot reach a running daemon.

For a new Docker installation, the standard smoke test is also useful:

```console
docker run --rm hello-world
```

This command pulls a small public image, starts a container, prints a confirmation, and removes the
container. It therefore checks more than `docker info`, but it requires network access on the first
run.

### 2. Check the Python SDK

Run the check through the same environment used by the analysis:

```console
uv run python -c "import docker; print(docker.from_env().ping())"
```

Without uv, use the environment's Python directly:

```console
python -c "import docker; print(docker.from_env().ping())"
```

The expected result is `True`. A successful Docker command followed by a failed Python check often
means the script is using a different environment or that Docker connection variables are set only
in one shell.

## How MGRP uses the container

When a {class}`~gerrytools.mgrp.RunContainer` context is entered, GerryTools:

1. connects to the daemon using the Docker SDK;
2. attempts to pull the image pinned by the installed GerryTools version;
3. falls back to an existing local copy if the pull is unavailable;
4. mounts the configured input, output, and log locations;
5. starts the sampler with container networking disabled; and
6. removes the temporary container when the context exits.

The image is large enough that the first pull can take several minutes. Later runs reuse the local
layers. You normally do not need to select or pull an image manually.

To confirm that an MGRP image is already available, run:

```console
docker image ls mgggdev/replicate
```

Do not substitute a moving `latest` tag for a published analysis. Record the GerryTools version and
the complete image reference when reproducibility matters. Passing a custom image to `RunContainer`
is intended for controlled development or a project that deliberately maintains its own image.

## File sharing and mounted paths

Runner configurations contain host paths that Docker exposes inside the container. A path can
exist from Python's perspective and still be unavailable to Docker Desktop if its parent directory
is not shared.

When diagnosing a mount problem:

- use absolute paths for the input, output, and log directories;
- confirm that every parent directory is shared with Docker Desktop;
- confirm that the current user can read the input and write to the output and log directories;
- try a small run from a local directory before using a network mount or cloud-synchronized folder;
- remember that Windows, macOS, and Linux containers may treat filename case differently.

Cloud-synchronized and network-mounted directories add another filesystem layer between the
sampler and Docker. They can work, but a small local run helps distinguish an MGRP configuration
problem from delayed synchronization, file locking, or mount performance.

## Resource planning

Containers share the host's CPU, memory, and disk. Before a production run:

- confirm Docker Desktop's CPU, memory, and disk limits;
- make sure the output filesystem and Docker image store both have free space;
- run a small number of steps using the production graph, columns, and mounts;
- inspect the log and output files before increasing the run size.

MGRP samplers can use substantial memory. If the operating system kills Docker or the container
stops without a useful sampler error, check Docker Desktop's resource limits and the host's free
memory.

## Common failures

`docker: command not found`
: Docker is not installed, or its CLI directory is not on `PATH`.

`Cannot connect to the Docker daemon` or a failed Python `ping()`
: Start Docker Desktop or the system Docker service. On Linux, also check socket permissions.

`Mounts denied`, `path is not shared`, or an input file missing only inside the container
: Add the parent directory to Docker Desktop's file-sharing settings and use an absolute path.

The image cannot be pulled
: Check network or registry access. GerryTools can use a matching local image when the pull fails,
but it cannot start if no local copy exists.

The container exits during a large run
: Inspect the configured log file first, then check memory, disk space, and Docker Desktop resource
limits before retrying.

Once these checks pass, continue with {doc}`the MGRP overview <../user/mgrp>` and choose the
runner-specific ReCom, Forest, or SMC guide.
