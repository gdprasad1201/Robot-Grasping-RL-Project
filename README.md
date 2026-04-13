# DexMobile PyBullet Project

This project uses PyBullet and reinforcement learning to train and evaluate models for robotic tasks.

## Environment Setup

### 1. Open terminal

Open a terminal and go to your project folder.

This step is the same on Windows, macOS, and Linux. Only the terminal app is different:

- **Windows:** Command Prompt, PowerShell, or Anaconda Prompt
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

### 2. Create conda environment

Create a new conda environment with Python 3.10.12:

```bash
conda create -n dexmobile python=3.10.12 -y
```

### 3. Activate environment

```bash
conda activate dexmobile
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install required libraries

```bash
pip install gym==0.26.1 gym-notices==0.1.0 gymnasium==1.2.3 Shimmy==2.0.0 numpy==1.26.4 pandas==2.3.3 pybullet==3.2.7 stable-baselines3==2.7.1 setuptools==80.9.0
```

### 6. Verify installation

```bash
pip show gym gymnasium Shimmy numpy pandas pybullet stable-baselines3 setuptools
```

## Running the Project

### 7. Run your project

```bash
python train.py
```

**Note:**

- Training may take several hours depending on your system
- Be patient while it runs
- To reduce training time, you can start by training on a single object or a limited number of objects

### 8. Evaluate your model

After training is complete, evaluate the trained model:

```bash
python evaluate.py
```

If your evaluation script has a different name, replace `evaluate.py` with the correct file name.

## Quick Version

```bash
conda create -n dexmobile python=3.10.12 -y
conda activate dexmobile
python -m pip install --upgrade pip
pip install gym==0.26.1 gym-notices==0.1.0 gymnasium==1.2.3 Shimmy==2.0.0 numpy==1.26.4 pandas==2.3.3 pybullet==3.2.7 stable-baselines3==2.7.1 setuptools==80.9.0
python train.py
python evaluate.py
```

## Notes

- This project uses both `gym` and `gymnasium` for compatibility
- `Shimmy` is included to bridge differences between APIs
- Make sure you are inside the correct conda environment before running scripts
