# DexMobile PyBullet Project

This project uses PyBullet and reinforcement learning to train and evaluate models for robotic tasks.

## Environment Setup

### 1. Open terminal

Open a terminal and go to your project folder.

This step is the same on Windows, macOS, and Linux. Only the terminal app is different:

- **Windows:** Command Prompt, PowerShell
- **macOS:** Terminal
- **Linux:** Terminal

Example:

```bash
cd /path/to/your/project
```

On Windows, it may look like this:

```bash
cd C:\path\to\your\project
```

### 2. Create virtual environment with [uv](https://docs.astral.sh/uv/getting-started/installation/)

Create a new uv environment with Python 3.10:

```bash
uv venv
```

### 3. Activate environment

On Windows:

```bash
.venv/Scripts/activate
```

On macOS and Linux:

```bash
source .venv/bin/activate
```

### 4. Install required libraries

```bash
uv sync --active
```

### 5. Verify installation

```bash
pip show gym gymnasium Shimmy numpy pandas pybullet stable-baselines3 setuptools
```

## Running the Project

### 6. Run your project and evaluate your model

```bash
./run_pipeline.sh
```

**Note:**

- Training may take several hours depending on your system
- Be patient while it runs
- To reduce training time, you can start by training on a single object or a limited number of objects

If your evaluation script has a different name, replace `evaluate.py` with the correct file name.

## Notes

- This project uses both `gym` and `gymnasium` for compatibility
- `Shimmy` is included to bridge differences between APIs
- Make sure you are inside the correct virtual environment before running scripts
