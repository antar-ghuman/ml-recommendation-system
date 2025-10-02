import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required = [
        'numpy', 'pandas', 'sklearn', 'surprise', 
        'fastapi', 'mlflow', 'torch'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0

def check_data():
    """Check if dataset exists"""
    data_path = Path("data/raw/ml-25m")
    
    required_files = ['ratings.csv', 'movies.csv', 'tags.csv']
    
    if not data_path.exists():
        print("❌ Dataset not found. Run: python scripts/download_data.py")
        return False
    
    for file in required_files:
        if (data_path / file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            return False
    
    return True

def check_structure():
    """Check if directory structure exists"""
    required_dirs = [
        'data/raw', 'data/processed', 'models', 
        'src/algorithms', 'src/evaluation', 'src/api',
        'notebooks', 'config', 'experiments'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("=" * 50)
    print("CHECKING DEPENDENCIES")
    print("=" * 50)
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    print("CHECKING DATA")
    print("=" * 50)
    data_ok = check_data()
    
    print("\n" + "=" * 50)
    print("CHECKING STRUCTURE")
    print("=" * 50)
    structure_ok = check_structure()
    
    print("\n" + "=" * 50)
    if deps_ok and data_ok and structure_ok:
        print("🎉 SETUP COMPLETE - Ready to build!")
    else:
        print("⚠️  Some issues found - fix them before proceeding")
        sys.exit(1)
        