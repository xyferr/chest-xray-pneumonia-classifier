"""Verify project setup before training."""
import sys
from pathlib import Path

def verify_setup():
    print("=" * 60)
    print("🔍 Verifying Project Setup")
    print("=" * 60)
    
    errors = []
    
    # Check Python version
    print(f"\n📌 Python Version: {sys.version}")
    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ required")
    
    # Check required packages
    print("\n📦 Checking Dependencies...")
    required = ['torch', 'torchvision', 'numpy', 'PIL', 'sklearn', 'tqdm', 'matplotlib', 'seaborn']
    for pkg in required:
        try:
            if pkg == 'PIL':
                import PIL
                print(f"  ✅ Pillow: {PIL.__version__}")
            elif pkg == 'sklearn':
                import sklearn
                print(f"  ✅ scikit-learn: {sklearn.__version__}")
            else:
                mod = __import__(pkg)
                ver = getattr(mod, '__version__', 'installed')
                print(f"  ✅ {pkg}: {ver}")
        except ImportError:
            print(f"  ❌ {pkg}: NOT FOUND")
            errors.append(f"Missing package: {pkg}")
    
    # Check CUDA
    print("\n🎮 GPU Status...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ CUDA Version: {torch.version.cuda}")
        else:
            print("  ⚠️ CUDA not available, will use CPU (slower)")
    except Exception as e:
        print(f"  ⚠️ Could not check CUDA: {e}")
    
    # Check dataset
    print("\n📁 Checking Dataset...")
    data_path = Path("data/chest_xray")
    if data_path.exists():
        for split in ['train', 'val', 'test']:
            split_path = data_path / split
            if split_path.exists():
                normal = len(list((split_path / "NORMAL").glob("*")))
                pneumonia = len(list((split_path / "PNEUMONIA").glob("*")))
                print(f"  ✅ {split}: {normal:,} normal, {pneumonia:,} pneumonia")
            else:
                print(f"  ❌ {split}: NOT FOUND")
                errors.append(f"Missing split: {split}")
    else:
        print("  ❌ Dataset not found at data/chest_xray/")
        errors.append("Dataset not found")
    
    # Check source files
    print("\n📄 Checking Source Files...")
    src_files = ['utils.py', 'dataset.py', 'models.py', 'train.py', 'eval.py']
    for f in src_files:
        fpath = Path(f"src/{f}")
        if fpath.exists():
            lines = len(fpath.read_text().splitlines())
            print(f"  ✅ src/{f}: {lines} lines")
        else:
            print(f"  ❌ src/{f}: NOT FOUND")
            errors.append(f"Missing file: src/{f}")
    
    # Create output directories
    print("\n📂 Creating Output Directories...")
    output_dirs = [
        'outputs/checkpoints',
        'outputs/tensorboard',
        'outputs/evaluation',
        'outputs/gradcam'
    ]
    for d in output_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {d}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ Setup has issues:")
        for e in errors:
            print(f"   - {e}")
        return False
    else:
        print("✅ All checks passed! Ready to train.")
        return True

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)